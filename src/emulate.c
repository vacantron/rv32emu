/*
 * rv32emu is freely redistributable under the MIT License. See the file
 * "LICENSE" for information on usage and redistribution of this file.
 */

#include <assert.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#if RV32_HAS(EXT_F)
#include <math.h>
#include "softfloat.h"
#endif /* RV32_HAS(EXT_F) */

#if RV32_HAS(GDBSTUB)
extern struct target_ops gdbstub_ops;
#endif

#include "decode.h"
#include "io.h"
#include "mpool.h"
#include "riscv.h"
#include "riscv_private.h"
#include "utils.h"

#if RV32_HAS(JIT)
#include "cache.h"
#include "jit.h"
#endif

/* Shortcuts for comparing each field of specified RISC-V instruction */
#define IF_insn(i, o) (i->opcode == rv_insn_##o)
#define IF_rd(i, r) (i->rd == rv_reg_##r)
#define IF_rs1(i, r) (i->rs1 == rv_reg_##r)
#define IF_rs2(i, r) (i->rs2 == rv_reg_##r)
#define IF_imm(i, v) (i->imm == v)

/* RISC-V trap code list */
/* clang-format off */
#define RV_TRAP_LIST                                                                               \
    IIF(RV32_HAS(EXT_C))(,                                                                         \
        _(insn_misaligned, 0)                           /* Instruction address misaligned */       \
    )                                                                                              \
    _(illegal_insn, 2)                                  /* Illegal instruction */                  \
    _(breakpoint, 3)                                    /* Breakpoint */                           \
    _(load_misaligned, 4)                               /* Load address misaligned */              \
    _(store_misaligned, 6)                              /* Store/AMO address misaligned */         \
    _(ecall_M, 11)                                      /* Environment call from M-mode */         \
    IIF(RV32_HAS(SYSTEM))(                                                                         \
        _(ecall_U, 8)                                   /* Environment call from U-mode */         \
        _(ecall_S, 9)                                   /* Environment call from S-mode */         \
        _(pagefault_insn, 12)                           /* Instruction page fault */               \
        _(pagefault_load, 13)                           /* Load page fault */                      \
        _(pagefault_store, 15)                          /* Store page fault */                     \
	_(supervisor_sw_intr, (1U << 31) | 1)           /* Supervisor software interrupt */        \
        _(supervisor_timer_intr, (1U << 31) | 5)        /* Supervisor timer interrupt */           \
        _(supervisor_external_intr, (1U << 31) | 9)     /* Supervisor external interrupt */        \
    )
/* clang-format on */

enum {
#define _(type, code) rv_trap_code_##type = code,
    RV_TRAP_LIST
#undef _
};

static void rv_trap_default_handler(riscv_t *rv)
{
    rv->csr_mepc += rv->compressed ? 2 : 4;
    rv->PC = rv->csr_mepc; /* mret */
}

#if RV32_HAS(SYSTEM)
static void trap_handler(riscv_t *rv);
#else
/* should not be called in non-SYSTEM mode since default trap handler is capable
 * to handle traps
 */
static void trap_handler(riscv_t *rv UNUSED) {}
#endif

bool can_trapped = false;
uint32_t satp_cnt;

/* When a trap occurs in M-mode, mtval is either initialized to zero or
 * populated with exception-specific details to assist software in managing
 * the trap. Otherwise, the implementation never modifies mtval, although
 * software can explicitly write to it. The hardware platform will define
 * which exceptions are required to informatively set mtval and which may
 * consistently set it to zero.
 *
 * When a hardware breakpoint is triggered or an exception like address
 * misalignment, access fault, or page fault occurs during an instruction
 * fetch, load, or store operation, mtval is updated with the virtual address
 * that caused the fault. In the case of an illegal instruction trap, mtval *
 * might be updated with the first XLEN or ILEN bits of the offending *
 * instruction. For all other traps, mtval is simply set to zero. However, it is
 * worth noting that a future standard could redefine how mtval is handled for
 * different types of traps.
 */
static jmp_buf env;
static jmp_buf nested_env;
#define TRAP_HANDLER_IMPL(type, code)                                         \
    static void rv_trap_##type(riscv_t *rv, uint32_t mtval)                   \
    {                                                                         \
        /* m/stvec (Machine/Supervisor Trap-Vector Base Address Register)     \
         * m/stvec[MXLEN-1:2]: vector base address                            \
         * m/stvec[1:0] : vector mode                                         \
         */                                                                   \
        uint32_t base;                                                        \
        uint32_t mode;                                                        \
        /* m/sepc  (Machine/Supervisor Exception Program Counter)             \
         * m/stval (Machine/Supervisor Trap Value Register)                   \
         * m/scause (Machine/Supervisor Cause Register): store exception code \
         * m/sstatus (Machine/Supervisor Status Register): keep track of and  \
         * controls the hart’s current operating state                      \
         */                                                                   \
        /* supervisor */                                                      \
        const uint32_t sstatus_sie =                                          \
            (rv->csr_sstatus & SSTATUS_SIE) >> SSTATUS_SIE_SHIFT;             \
        rv->csr_sstatus |= (sstatus_sie << SSTATUS_SPIE_SHIFT);               \
        rv->csr_sstatus &= ~(SSTATUS_SIE);                                    \
        rv->csr_sstatus |= (rv->priv_mode << SSTATUS_SPP_SHIFT);              \
        rv->priv_mode = RV_PRIV_S_MODE;                                       \
        base = rv->csr_stvec & ~0x3;                                          \
        mode = rv->csr_stvec & 0x3;                                           \
        rv->csr_sepc = rv->PC;                                                \
        rv->csr_stval = mtval;                                                \
        rv->csr_scause = code;                                                \
        switch (mode) {                                                       \
        /* DIRECT: All traps set PC to base */                                \
        case 0:                                                               \
            rv->PC = base;                                                    \
            break;                                                            \
        /* VECTORED: Asynchronous traps set PC to base + 4 * code */          \
        case 1:                                                               \
            /* MSB of code is used to indicate whether the trap is interrupt  \
             * or exception, so it is not considered as the 'real' code */    \
            rv->PC = base + 4 * (code & MASK(31));                            \
            break;                                                            \
        }                                                                     \
        /* block escaping for trap handling */                                \
        if (satp_cnt >= 2 && rv->is_trapped) {                                \
            trap_handler(rv);                                                 \
        }                                                                     \
    }

/* RISC-V trap handlers */
#define _(type, code) TRAP_HANDLER_IMPL(type, code)
RV_TRAP_LIST
#undef _

/* wrap load/store and insn misaligned handler
 * @mask_or_pc: mask for load/store and pc for insn misaligned handler.
 * @type: type of misaligned handler
 * @compress: compressed instruction or not
 * @IO: whether the misaligned handler is for load/store or insn.
 */
#define RV_EXC_MISALIGN_HANDLER(mask_or_pc, type, compress, IO)       \
    IIF(IO)                                                           \
    (if (!PRIV(rv)->allow_misalign && unlikely(addr & (mask_or_pc))), \
     if (unlikely(insn_is_misaligned(PC))))                           \
    {                                                                 \
        rv->compressed = compress;                                    \
        rv->csr_cycle = cycle;                                        \
        rv->PC = PC;                                                  \
        if (satp_cnt >= 2)                                            \
            rv->is_trapped = true;                                    \
        rv_trap_##type##_misaligned(rv, IIF(IO)(addr, mask_or_pc));   \
        return false;                                                 \
    }


/* get current time in microsecnds and update csr_time register */
static inline void update_time(riscv_t *rv)
{
    struct timeval tv;
    rv_gettimeofday(&tv);

    uint64_t t = (uint64_t) tv.tv_sec * 1e6 + (uint32_t) tv.tv_usec;
    rv->csr_time[0] = t & 0xFFFFFFFF;
    rv->csr_time[1] = t >> 32;
}

static inline void get_time_now(struct timeval *tv)
{
    rv_gettimeofday(tv);
}

#if RV32_HAS(Zicsr)
/* get a pointer to a CSR */
static uint32_t *csr_get_ptr(riscv_t *rv, uint32_t csr)
{
    /* csr & 0xFFF prevent sign-extension in decode stage */
    switch (csr & 0xFFF) {
    case CSR_MSTATUS: /* Machine Status */
        return (uint32_t *) (&rv->csr_mstatus);
    case CSR_MTVEC: /* Machine Trap Handler */
        return (uint32_t *) (&rv->csr_mtvec);
    case CSR_MISA: /* Machine ISA and Extensions */
        return (uint32_t *) (&rv->csr_misa);

    /* Machine Trap Handling */
    case CSR_MEDELEG: /* Machine Exception Delegation Register */
        return (uint32_t *) (&rv->csr_medeleg);
    case CSR_MIDELEG: /* Machine Interrupt Delegation Register */
        return (uint32_t *) (&rv->csr_mideleg);
    case CSR_MSCRATCH: /* Machine Scratch Register */
        return (uint32_t *) (&rv->csr_mscratch);
    case CSR_MEPC: /* Machine Exception Program Counter */
        return (uint32_t *) (&rv->csr_mepc);
    case CSR_MCAUSE: /* Machine Exception Cause */
        return (uint32_t *) (&rv->csr_mcause);
    case CSR_MTVAL: /* Machine Trap Value */
        return (uint32_t *) (&rv->csr_mtval);
    case CSR_MIP: /* Machine Interrupt Pending */
        return (uint32_t *) (&rv->csr_mip);

    /* Machine Counter/Timers */
    case CSR_CYCLE: /* Cycle counter for RDCYCLE instruction */
        return (uint32_t *) &rv->csr_cycle;
    case CSR_CYCLEH: /* Upper 32 bits of cycle */
        return &((uint32_t *) &rv->csr_cycle)[1];

    /* TIME/TIMEH - very roughly about 1 ms per tick */
    case CSR_TIME: /* Timer for RDTIME instruction */
        update_time(rv);
        return &rv->csr_time[0];
    case CSR_TIMEH: /* Upper 32 bits of time */
        update_time(rv);
        return &rv->csr_time[1];
    case CSR_INSTRET: /* Number of Instructions Retired Counter */
        return (uint32_t *) (&rv->csr_cycle);
#if RV32_HAS(EXT_F)
    case CSR_FFLAGS:
        return (uint32_t *) (&rv->csr_fcsr);
    case CSR_FCSR:
        return (uint32_t *) (&rv->csr_fcsr);
#endif
    case CSR_SSTATUS:
        return (uint32_t *) (&rv->csr_sstatus);
    case CSR_SIE:
        return (uint32_t *) (&rv->csr_sie);
    case CSR_STVEC:
        return (uint32_t *) (&rv->csr_stvec);
    case CSR_SCOUNTEREN:
        return (uint32_t *) (&rv->csr_scounteren);
    case CSR_SSCRATCH:
        return (uint32_t *) (&rv->csr_sscratch);
    case CSR_SEPC:
        return (uint32_t *) (&rv->csr_sepc);
    case CSR_SCAUSE:
        return (uint32_t *) (&rv->csr_scause);
    case CSR_STVAL:
        return (uint32_t *) (&rv->csr_stval);
    case CSR_SIP:
        return (uint32_t *) (&rv->csr_sip);
    case CSR_SATP:
        return (uint32_t *) (&rv->csr_satp);
    default:
        return NULL;
    }
}

/* CSRRW (Atomic Read/Write CSR) instruction atomically swaps values in the
 * CSRs and integer registers. CSRRW reads the old value of the CSR,
 * zero-extends the value to XLEN bits, and then writes it to register rd.
 * The initial value in rs1 is written to the CSR.
 * If rd == x0, then the instruction shall not read the CSR and shall not cause
 * any of the side effects that might occur on a CSR read.
 */
static uint32_t csr_csrrw(riscv_t *rv, uint32_t csr, uint32_t val)
{
    uint32_t *c = csr_get_ptr(rv, csr);
    if (!c)
        return 0;

    uint32_t out = *c;
#if RV32_HAS(EXT_F)
    if (csr == CSR_FFLAGS)
        out &= FFLAG_MASK;
#endif

    if (c == &rv->csr_satp) {
        const uint8_t mode_sv32 = val >> 31;
        val &= ~(MASK(9) << 22); /* disable ASID allocator */
        if (mode_sv32)
            *c = val;
        else        /* bare mode */
            *c = 0; /* virtual mem addr maps to same
                     * physical mem addr directly
                     */
        satp_cnt++;
    } else {
        *c = val;
    }

    return out;
}

/* perform csrrs (atomic read and set) */
static uint32_t csr_csrrs(riscv_t *rv, uint32_t csr, uint32_t val)
{
    uint32_t *c = csr_get_ptr(rv, csr);
    if (!c)
        return 0;

    uint32_t out = *c;
#if RV32_HAS(EXT_F)
    if (csr == CSR_FFLAGS)
        out &= FFLAG_MASK;
#endif

    *c |= val;

    return out;
}

/* perform csrrc (atomic read and clear)
 * Read old value of CSR, zero-extend to XLEN bits, write to rd.
 * Read value from rs1, use as bit mask to clear bits in CSR.
 */
static uint32_t csr_csrrc(riscv_t *rv, uint32_t csr, uint32_t val)
{
    uint32_t *c = csr_get_ptr(rv, csr);
    if (!c)
        return 0;

    uint32_t out = *c;
#if RV32_HAS(EXT_F)
    if (csr == CSR_FFLAGS)
        out &= FFLAG_MASK;
#endif

    *c &= ~val;

    return out;
}
#endif

#if RV32_HAS(GDBSTUB)
void rv_debug(riscv_t *rv)
{
    if (!gdbstub_init(&rv->gdbstub, &gdbstub_ops,
                      (arch_info_t){
                          .reg_num = 33,
                          .reg_byte = 4,
                          .target_desc = TARGET_RV32,
                      },
                      GDBSTUB_COMM)) {
        return;
    }

    rv->debug_mode = true;
    rv->breakpoint_map = breakpoint_map_new();
    rv->is_interrupted = false;

    if (!gdbstub_run(&rv->gdbstub, (void *) rv))
        return;

    breakpoint_map_destroy(rv->breakpoint_map);
    gdbstub_close(&rv->gdbstub);
}
#endif /* RV32_HAS(GDBSTUB) */

#if !RV32_HAS(JIT)
/* hash function for the block map */
HASH_FUNC_IMPL(map_hash, BLOCK_MAP_CAPACITY_BITS, 1 << BLOCK_MAP_CAPACITY_BITS)
#endif

/* allocate a basic block */
static block_t *block_alloc(riscv_t *rv)
{
    block_t *block = mpool_alloc(rv->block_mp);
    assert(block);
    block->n_insn = 0;
#if RV32_HAS(JIT)
    block->translatable = true;
    block->hot = false;
    block->hot2 = false;
    block->has_loops = false;
    block->n_invoke = 0;
    INIT_LIST_HEAD(&block->list);
#if RV32_HAS(T2C)
    block->compiled = false;
#endif
#endif
    return block;
}

#if !RV32_HAS(JIT)
/* insert a block into block map */
static void block_insert(block_map_t *map, const block_t *block)
{
    assert(map && block);
    const uint32_t mask = map->block_capacity - 1;
    uint32_t index = map_hash(block->pc_start);

    /* insert into the block map */
    for (;; index++) {
        if (!map->map[index & mask]) {
            map->map[index & mask] = (block_t *) block;
            break;
        }
    }
    map->size++;
}

/* try to locate an already translated block in the block map */
static block_t *block_find(const block_map_t *map, const uint32_t addr)
{
    assert(map);
    uint32_t index = map_hash(addr);
    const uint32_t mask = map->block_capacity - 1;

    /* find block in block map */
    for (;; index++) {
        block_t *block = map->map[index & mask];
        if (!block)
            return NULL;

        if (block->pc_start == addr)
            return block;
    }
    return NULL;
}
#endif

#if !RV32_HAS(EXT_C)
FORCE_INLINE bool insn_is_misaligned(uint32_t pc)
{
    return pc & 0x3;
}
#endif

/* instruction length information for each RISC-V instruction */
enum {
#define _(inst, can_branch, insn_len, translatable, reg_mask) \
    __rv_insn_##inst##_len = insn_len,
    RV_INSN_LIST
#undef _
};

/* can-branch information for each RISC-V instruction */
enum {
#define _(inst, can_branch, insn_len, translatable, reg_mask) \
    __rv_insn_##inst##_canbranch = can_branch,
    RV_INSN_LIST
#undef _
};

#if RV32_HAS(GDBSTUB)
#define RVOP_NO_NEXT(ir) (!ir->next | rv->debug_mode | rv->is_trapped)
#else
#define RVOP_NO_NEXT(ir) (!ir->next | rv->is_trapped)
#endif

/* record whether the branch is taken or not during emulation */
static bool is_branch_taken = false;

/* record the program counter of the previous block */
static uint32_t last_pc = 0;

#if RV32_HAS(JIT)
static set_t pc_set;
static bool has_loops = false;
#endif

static void emu_update_uart_interrupts(riscv_t *rv);
static uint32_t peripheral_update_ctr = 64;

/* Interpreter-based execution path */
#define RVOP(inst, code, asm)                                         \
    static bool do_##inst(riscv_t *rv, rv_insn_t *ir, uint64_t cycle, \
                          uint32_t PC)                                \
    {                                                                 \
        cycle++;                                                      \
        code;                                                         \
    nextop:                                                           \
        /* If store/load page fault occurs, stop the execution */     \
        /* since PC is updated in rv_trap_xxx */                      \
        PC += __rv_insn_##inst##_len;                                 \
        if (unlikely(RVOP_NO_NEXT(ir))) {                             \
            goto end_op;                                              \
        }                                                             \
        const rv_insn_t *next = ir->next;                             \
        MUST_TAIL return next->impl(rv, next, cycle, PC);             \
    end_op:                                                           \
        rv->csr_cycle = cycle;                                        \
        rv->PC = PC;                                                  \
        return true;                                                  \
    }

#include "rv32_template.c"
#undef RVOP

/* multiple LUI */
static bool do_fuse1(riscv_t *rv, rv_insn_t *ir, uint64_t cycle, uint32_t PC)
{
    cycle += ir->imm2;
    opcode_fuse_t *fuse = ir->fuse;
    for (int i = 0; i < ir->imm2; i++)
        rv->X[fuse[i].rd] = fuse[i].imm;
    PC += ir->imm2 * 4;
    if ((PC >> 20) == 0x957 || ((rv->PC >> 20) == 0x957)) {
        // exit(1);
    }
    if (unlikely(RVOP_NO_NEXT(ir))) {
        rv->csr_cycle = cycle;
        rv->PC = PC;
        return true;
    }
    const rv_insn_t *next = ir->next;
    MUST_TAIL return next->impl(rv, next, cycle, PC);
}

/* LUI + ADD */
static bool do_fuse2(riscv_t *rv, rv_insn_t *ir, uint64_t cycle, uint32_t PC)
{
    cycle += 2;
    rv->X[ir->rd] = ir->imm;
    rv->X[ir->rs2] = rv->X[ir->rd] + rv->X[ir->rs1];
    PC += 8;
    if ((PC >> 20) == 0x957 || ((rv->PC >> 20) == 0x957)) {
        // exit(1);
    }
    if (unlikely(RVOP_NO_NEXT(ir))) {
        rv->csr_cycle = cycle;
        rv->PC = PC;
        return true;
    }
    const rv_insn_t *next = ir->next;
    MUST_TAIL return next->impl(rv, next, cycle, PC);
}

/* multiple SW */
static bool do_fuse3(riscv_t *rv, rv_insn_t *ir, uint64_t cycle, uint32_t PC)
{
    cycle += ir->imm2;
    opcode_fuse_t *fuse = ir->fuse;
    /* The memory addresses of the sw instructions are contiguous, thus only
     * the first SW instruction needs to be checked to determine if its memory
     * address is misaligned or if the memory chunk does not exist.
     */
    for (int i = 0; i < ir->imm2; i++) {
        uint32_t addr = rv->X[fuse[i].rs1] + fuse[i].imm;
        RV_EXC_MISALIGN_HANDLER(3, store, false, 1);
        rv->io.mem_write_w(rv, addr, rv->X[fuse[i].rs2]);
    }
    PC += ir->imm2 * 4;
    if (unlikely(RVOP_NO_NEXT(ir))) {
        rv->csr_cycle = cycle;
        rv->PC = PC;
        return true;
    }
    const rv_insn_t *next = ir->next;
    MUST_TAIL return next->impl(rv, next, cycle, PC);
}

/* multiple LW */
static bool do_fuse4(riscv_t *rv, rv_insn_t *ir, uint64_t cycle, uint32_t PC)
{
    cycle += ir->imm2;
    opcode_fuse_t *fuse = ir->fuse;
    /* The memory addresses of the lw instructions are contiguous, therefore
     * only the first LW instruction needs to be checked to determine if its
     * memory address is misaligned or if the memory chunk does not exist.
     */
    for (int i = 0; i < ir->imm2; i++) {
        uint32_t addr = rv->X[fuse[i].rs1] + fuse[i].imm;
        RV_EXC_MISALIGN_HANDLER(3, load, false, 1);
        rv->X[fuse[i].rd] = rv->io.mem_read_w(rv, addr);
    }
    PC += ir->imm2 * 4;
    if (unlikely(RVOP_NO_NEXT(ir))) {
        rv->csr_cycle = cycle;
        rv->PC = PC;
        return true;
    }
    const rv_insn_t *next = ir->next;
    MUST_TAIL return next->impl(rv, next, cycle, PC);
}

/* multiple shift immediate */
static bool do_fuse5(riscv_t *rv,
                     const rv_insn_t *ir,
                     uint64_t cycle,
                     uint32_t PC)
{
    cycle += ir->imm2;
    opcode_fuse_t *fuse = ir->fuse;
    for (int i = 0; i < ir->imm2; i++)
        shift_func(rv, (const rv_insn_t *) (&fuse[i]));
    PC += ir->imm2 * 4;
    if ((PC >> 20) == 0x957 || ((rv->PC >> 20) == 0x957)) {
        // exit(1);
    }
    if (unlikely(RVOP_NO_NEXT(ir))) {
        rv->csr_cycle = cycle;
        rv->PC = PC;
        return true;
    }
    const rv_insn_t *next = ir->next;
    MUST_TAIL return next->impl(rv, next, cycle, PC);
}

/* clang-format off */
static const void *dispatch_table[] = {
    /* RV32 instructions */
#define _(inst, can_branch, insn_len, translatable, reg_mask) [rv_insn_##inst] = do_##inst,
    RV_INSN_LIST
#undef _
    /* Macro operation fusion instructions */
#define _(inst) [rv_insn_##inst] = do_##inst,
    FUSE_INSN_LIST
#undef _
};
/* clang-format on */

FORCE_INLINE bool insn_is_branch(uint8_t opcode)
{
    switch (opcode) {
#define _(inst, can_branch, insn_len, translatable, reg_mask) \
    IIF(can_branch)                                           \
    (case rv_insn_##inst:, )
        RV_INSN_LIST
#undef _
        return true;
    }
    return false;
}

#if RV32_HAS(JIT)
FORCE_INLINE bool insn_is_translatable(uint8_t opcode)
{
    switch (opcode) {
#define _(inst, can_branch, insn_len, translatable, reg_mask) \
    IIF(translatable)                                         \
    (case rv_insn_##inst:, )
        RV_INSN_LIST
#undef _
        return true;
    }
    return false;
}
#endif

FORCE_INLINE bool insn_is_unconditional_branch(uint8_t opcode)
{
    switch (opcode) {
    case rv_insn_ecall:
    case rv_insn_ebreak:
    case rv_insn_jal:
    case rv_insn_jalr:
    case rv_insn_sret:
    case rv_insn_mret:
#if RV32_HAS(EXT_C)
    case rv_insn_cj:
    case rv_insn_cjalr:
    case rv_insn_cjal:
    case rv_insn_cjr:
    case rv_insn_cebreak:
#endif
        return true;
    }
    return false;
}
int found = false;

static void block_translate(riscv_t *rv, block_t *block)
{
    block->pc_start = block->pc_end = rv->PC;

    rv_insn_t *prev_ir = NULL;
    rv_insn_t *ir = mpool_alloc(rv->block_ir_mp);
    block->ir_head = ir;

    /* translate the basic block */
    while (true) {
        memset(ir, 0, sizeof(rv_insn_t));

        if (prev_ir)
            prev_ir->next = ir;

        /* fetch the next instruction */
        uint32_t insn = rv->io.mem_ifetch(rv, block->pc_end);
        assert(insn);

        /* decode the instruction */
        if (!rv_decode(ir, insn)) {
            rv->compressed = is_compressed(insn);
            rv_trap_illegal_insn(rv, insn);
            break;
        }
        ir->impl = dispatch_table[ir->opcode];
        ir->pc = block->pc_end; /* compute the end of pc */
        block->pc_end += is_compressed(insn) ? 2 : 4;
        block->n_insn++;
        prev_ir = ir;
#if RV32_HAS(JIT)
        if (!insn_is_translatable(ir->opcode))
            block->translatable = false;
#endif
        /* stop on branch */
        if (insn_is_branch(ir->opcode)) {
            if (ir->opcode == rv_insn_jalr
#if RV32_HAS(EXT_C)
                || ir->opcode == rv_insn_cjalr || ir->opcode == rv_insn_cjr
#endif
            ) {
                ir->branch_table = calloc(1, sizeof(branch_history_table_t));
                assert(ir->branch_table);
                memset(ir->branch_table->PC, -1,
                       sizeof(uint32_t) * HISTORY_SIZE);
            }
            break;
        }

        ir = mpool_alloc(rv->block_ir_mp);
    }

    assert(prev_ir);
    block->ir_tail = prev_ir;
    block->ir_tail->next = NULL;
}

#define COMBINE_MEM_OPS(RW)                                       \
    next_ir = ir->next;                                           \
    count = 1;                                                    \
    while (1) {                                                   \
        if (next_ir->opcode != IIF(RW)(rv_insn_lw, rv_insn_sw))   \
            break;                                                \
        count++;                                                  \
        if (!next_ir->next)                                       \
            break;                                                \
        next_ir = next_ir->next;                                  \
    }                                                             \
    if (count > 1) {                                              \
        ir->opcode = IIF(RW)(rv_insn_fuse4, rv_insn_fuse3);       \
        ir->fuse = malloc(count * sizeof(opcode_fuse_t));         \
        assert(ir->fuse);                                         \
        ir->imm2 = count;                                         \
        memcpy(ir->fuse, ir, sizeof(opcode_fuse_t));              \
        ir->impl = dispatch_table[ir->opcode];                    \
        next_ir = ir->next;                                       \
        for (int j = 1; j < count; j++, next_ir = next_ir->next)  \
            memcpy(ir->fuse + j, next_ir, sizeof(opcode_fuse_t)); \
        remove_next_nth_ir(rv, ir, block, count - 1);             \
    }

static inline void remove_next_nth_ir(const riscv_t *rv,
                                      rv_insn_t *ir,
                                      block_t *block,
                                      uint8_t n)
{
    for (uint8_t i = 0; i < n; i++) {
        rv_insn_t *next = ir->next;
        ir->next = ir->next->next;
        mpool_free(rv->block_ir_mp, next);
    }
    if (!ir->next)
        block->ir_tail = ir;
    block->n_insn -= n;
}

/* Check if instructions in a block match a specific pattern. If they do,
 * rewrite them as fused instructions.
 *
 * Strategies are being devised to increase the number of instructions that
 * match the pattern, including possible instruction reordering.
 */
static void match_pattern(riscv_t *rv, block_t *block)
{
    uint32_t i;
    rv_insn_t *ir;
    for (i = 0, ir = block->ir_head; i < block->n_insn - 1;
         i++, ir = ir->next) {
        assert(ir);
        rv_insn_t *next_ir = NULL;
        int32_t count = 0;
        switch (ir->opcode) {
        case rv_insn_lui:
            next_ir = ir->next;
            switch (next_ir->opcode) {
            case rv_insn_add:
                if (ir->rd == next_ir->rs2 || ir->rd == next_ir->rs1) {
                    ir->opcode = rv_insn_fuse2;
                    ir->rs2 = next_ir->rd;
                    if (ir->rd == next_ir->rs2)
                        ir->rs1 = next_ir->rs1;
                    else
                        ir->rs1 = next_ir->rs2;
                    ir->impl = dispatch_table[ir->opcode];
                    remove_next_nth_ir(rv, ir, block, 1);
                }
                break;
            case rv_insn_lui:
                count = 1;
                while (1) {
                    if (!IF_insn(next_ir, lui))
                        break;
                    count++;
                    if (!next_ir->next)
                        break;
                    next_ir = next_ir->next;
                }
                if (count > 1) {
                    ir->opcode = rv_insn_fuse1;
                    ir->fuse = malloc(count * sizeof(opcode_fuse_t));
                    assert(ir->fuse);
                    ir->imm2 = count;
                    memcpy(ir->fuse, ir, sizeof(opcode_fuse_t));
                    ir->impl = dispatch_table[ir->opcode];
                    next_ir = ir->next;
                    for (int j = 1; j < count; j++, next_ir = next_ir->next)
                        memcpy(ir->fuse + j, next_ir, sizeof(opcode_fuse_t));
                    remove_next_nth_ir(rv, ir, block, count - 1);
                }
                break;
            }
            break;
        /* If the memory addresses of a sequence of store or load instructions
         * are contiguous, combine these instructions.
         */
        case rv_insn_sw:
            COMBINE_MEM_OPS(0);
            break;
        case rv_insn_lw:
            COMBINE_MEM_OPS(1);
            break;
            /* TODO: mixture of SW and LW */
            /* TODO: reorder insturction to match pattern */
        case rv_insn_slli:
        case rv_insn_srli:
        case rv_insn_srai:
            count = 1;
            next_ir = ir->next;
            while (1) {
                if (!IF_insn(next_ir, slli) && !IF_insn(next_ir, srli) &&
                    !IF_insn(next_ir, srai))
                    break;
                count++;
                if (!next_ir->next)
                    break;
                next_ir = next_ir->next;
            }
            if (count > 1) {
                ir->fuse = malloc(count * sizeof(opcode_fuse_t));
                assert(ir->fuse);
                memcpy(ir->fuse, ir, sizeof(opcode_fuse_t));
                ir->opcode = rv_insn_fuse5;
                ir->imm2 = count;
                ir->impl = dispatch_table[ir->opcode];
                next_ir = ir->next;
                for (int j = 1; j < count; j++, next_ir = next_ir->next)
                    memcpy(ir->fuse + j, next_ir, sizeof(opcode_fuse_t));
                remove_next_nth_ir(rv, ir, block, count - 1);
            }
            break;
        }
    }
}

typedef struct {
    bool is_constant[N_RV_REGS];
    uint32_t const_val[N_RV_REGS];
} constopt_info_t;

#define CONSTOPT(inst, code)                                  \
    static void constopt_##inst(rv_insn_t *ir UNUSED,         \
                                constopt_info_t *info UNUSED) \
    {                                                         \
        code;                                                 \
    }

#include "rv32_constopt.c"
static const void *constopt_table[] = {
#define _(inst, can_branch, insn_len, translatable, reg_mask) \
    [rv_insn_##inst] = constopt_##inst,
    RV_INSN_LIST
#undef _
};
#undef CONSTOPT

typedef void (*constopt_func_t)(rv_insn_t *, constopt_info_t *);
static void optimize_constant(riscv_t *rv UNUSED, block_t *block)
{
    constopt_info_t info = {.is_constant[0] = true};
    assert(rv->X[0] == 0);

    uint32_t i;
    rv_insn_t *ir;
    for (i = 0, ir = block->ir_head; i < block->n_insn; i++, ir = ir->next)
        ((constopt_func_t) constopt_table[ir->opcode])(ir, &info);
}

static block_t *prev = NULL;
static block_t *block_find_or_translate(riscv_t *rv)
{
#if !RV32_HAS(JIT)
    block_map_t *map = &rv->block_map;
    /* lookup the next block in the block map */
    block_t *next = block_find(map, rv->PC);
#else
    /* lookup the next block in the block cache */
    block_t *next = (block_t *) cache_get(rv->block_cache, rv->PC, true);
#endif

    if (!next) {
#if !RV32_HAS(JIT)
        if (map->size * 1.25 > map->block_capacity) {
            block_map_clear(rv);
            prev = NULL;
        }
#endif
        /* allocate a new block */
        next = block_alloc(rv);
        block_translate(rv, next);

        optimize_constant(rv, next);
#if RV32_HAS(GDBSTUB)
        if (likely(!rv->debug_mode))
#endif
            /* macro operation fusion */
            match_pattern(rv, next);

#if !RV32_HAS(JIT)
        /* insert the block into block map */
        block_insert(&rv->block_map, next);
#else
        /* insert the block into block cache */
        block_t *delete_target = cache_put(rv->block_cache, rv->PC, &(*next));
        if (delete_target) {
            if (prev == delete_target)
                prev = NULL;
            chain_entry_t *entry, *safe;
            /* correctly remove deleted block from its chained block */
            rv_insn_t *taken = delete_target->ir_tail->branch_taken,
                      *untaken = delete_target->ir_tail->branch_untaken;
            if (taken && taken->pc != delete_target->pc_start) {
                block_t *target = cache_get(rv->block_cache, taken->pc, false);
                bool flag = false;
                list_for_each_entry_safe (entry, safe, &target->list, list) {
                    if (entry->block == delete_target) {
                        list_del_init(&entry->list);
                        mpool_free(rv->chain_entry_mp, entry);
                        flag = true;
                    }
                }
                assert(flag);
            }
            if (untaken && untaken->pc != delete_target->pc_start) {
                block_t *target =
                    cache_get(rv->block_cache, untaken->pc, false);
                assert(target);
                bool flag = false;
                list_for_each_entry_safe (entry, safe, &target->list, list) {
                    if (entry->block == delete_target) {
                        list_del_init(&entry->list);
                        mpool_free(rv->chain_entry_mp, entry);
                        flag = true;
                    }
                }
                assert(flag);
            }
            /* correctly remove deleted block from the block chained to it */
            list_for_each_entry_safe (entry, safe, &delete_target->list, list) {
                if (entry->block == delete_target)
                    continue;
                rv_insn_t *target = entry->block->ir_tail;
                if (target->branch_taken == delete_target->ir_head)
                    target->branch_taken = NULL;
                else if (target->branch_untaken == delete_target->ir_head)
                    target->branch_untaken = NULL;
                mpool_free(rv->chain_entry_mp, entry);
            }
            /* free deleted block */
            uint32_t idx;
            rv_insn_t *ir, *next;
            for (idx = 0, ir = delete_target->ir_head;
                 idx < delete_target->n_insn; idx++, ir = next) {
                free(ir->fuse);
                next = ir->next;
                mpool_free(rv->block_ir_mp, ir);
            }
            mpool_free(rv->block_mp, delete_target);
        }
#endif
    }

    return next;
}

#if RV32_HAS(JIT)
static bool runtime_profiler(riscv_t *rv, block_t *block)
{
    /* Based on our observations, a significant number of true hotspots are
     * characterized by high usage frequency and including loop. Consequently,
     * we posit that our profiler could effectively identify hotspots using
     * three key indicators.
     */
    uint32_t freq = cache_freq(rv->block_cache, block->pc_start);
    /* To profile a block after chaining, it must first be executed. */
    if (unlikely(freq >= 2 && block->has_loops))
        return true;
    /* using frequency exceeds predetermined threshold */
    if (unlikely(freq == THRESHOLD))
        return true;
    return false;
}
#endif

void plic_update_interrupts(riscv_t *rv)
{
    vm_attr_t *attr = PRIV(rv);
    plic_t *plic = attr->plic;

    /* Update pending interrupts */
    plic->ip |= plic->active & ~plic->masked;
    plic->masked |= plic->active;
    /* Send interrupt to target */
    if (plic->ip & plic->ie) {
        rv->csr_sip |= SIP_SEIP;
    } else
        rv->csr_sip &= ~SIP_SEIP;
}

static uint32_t plic_read(riscv_t *rv, const uint32_t addr)
{
    vm_attr_t *attr = PRIV(rv);
    plic_t *plic = attr->plic;

    /* no priority support: source priority hardwired to 1 */
    if (1 <= addr && addr <= 31)
        return 0;

    uint32_t plic_read_val = 0;

    switch (addr) {
    case 0x400:
        plic_read_val = plic->ip;
        break;
    case 0x800:
        plic_read_val = plic->ie;
        break;
    case 0x80000:
        /* no priority support: target priority threshold hardwired to 0 */
        plic_read_val = 0;
        break;
    case 0x80001:
        /* claim */
        {
            uint32_t intr_candidate = plic->ip & plic->ie;
            if (intr_candidate) {
                plic_read_val = ilog2(intr_candidate);
                plic->ip &= ~(1U << (plic_read_val));
            }
            break;
        }
    default:
        return 0;
    }

    return plic_read_val;
}

static void plic_write(riscv_t *rv, const uint32_t addr, uint32_t value)
{
    vm_attr_t *attr = PRIV(rv);
    plic_t *plic = attr->plic;

    /* no priority support: source priority hardwired to 1 */
    if (1 <= addr && addr <= 31)
        return;

    switch (addr) {
    case 0x800:
        plic->ie = (value & ~1);
        break;
    case 0x80000:
        /* no priority support: target priority threshold hardwired to 0 */
        break;
    case 0x80001:
        /* completion */
        if (plic->ie & (1U << value))
            plic->masked &= ~(1U << value);
        break;
    default:
        break;
    }

    return;
}

static bool rv_has_plic_trap(riscv_t *rv)
{
    return ((rv->csr_sstatus & SSTATUS_SIE || !rv->priv_mode) &&
            (rv->csr_sip & rv->csr_sie));
}

void rv_step(void *arg)
{
    assert(arg);
    riscv_t *rv = arg;

    vm_attr_t *attr = PRIV(rv);
    uint32_t cycles = attr->cycle_per_step;

    /* find or translate a block for starting PC */
    const uint64_t cycles_target = rv->csr_cycle + cycles;

    /* now time */
    struct timeval tv;

    /* loop until hitting the cycle target */
    while (rv->csr_cycle < cycles_target && !rv->halt) {
        /* check for any interrupt after every block emulation */

        if (peripheral_update_ctr-- == 0) {
            peripheral_update_ctr = 64;

            u8250_check_ready(PRIV(rv)->uart);
            if (PRIV(rv)->uart->in_ready)
                emu_update_uart_interrupts(rv);
        }

        get_time_now(&tv);
        uint64_t t = (uint64_t) (tv.tv_sec * 1e6) + (uint32_t) tv.tv_usec;

        if (t > attr->timer) {
            rv->csr_sip |= RV_INT_STI;
        } else {
            rv->csr_sip &= ~RV_INT_STI;
        }

        if (rv_has_plic_trap(rv)) {
            if (satp_cnt >= 2) {
                rv->is_trapped = true;
            }
            uint32_t intr_applicable = rv->csr_sip & rv->csr_sie;
            uint8_t intr_idx = ilog2(intr_applicable);
            switch (intr_idx) {
            case 1:
                rv_trap_supervisor_sw_intr(rv, 0);
                break;
            case 5:
                rv_trap_supervisor_timer_intr(rv, 0);
                break;
            case 9:
                rv_trap_supervisor_external_intr(rv, 0);
                break;
            default:
                break;
            }
        }

        if (prev && prev->pc_start != last_pc) {
            /* update previous block */
            prev = block_find(&rv->block_map, last_pc);
        }
        /* lookup the next block in block map or translate a new block,
         * and move onto the next block.
         */
        block_t *block = block_find_or_translate(rv);
        /* by now, a block should be available */
        assert(block);
        // assert(block->n_insn == 1);

        /* After emulating the previous block, it is determined whether the
         * branch is taken or not. The IR array of the current block is then
         * assigned to either the branch_taken or branch_untaken pointer of
         * the previous block.
         */

        if (prev) {
            rv_insn_t *last_ir = prev->ir_tail;
            /* chain block */
            if (!insn_is_unconditional_branch(last_ir->opcode)) {
                if (is_branch_taken && !last_ir->branch_taken) {
                    last_ir->branch_taken = block->ir_head;
                } else if (!is_branch_taken && !last_ir->branch_untaken) {
                    last_ir->branch_untaken = block->ir_head;
                }
            } else if (IF_insn(last_ir, jal)
#if RV32_HAS(EXT_C)
                       || IF_insn(last_ir, cj) || IF_insn(last_ir, cjal)
#endif
            ) {
                if (!last_ir->branch_taken) {
                    last_ir->branch_taken = block->ir_head;
                }
            }
        }
        last_pc = rv->PC;

        /* execute the block by interpreter */
        const rv_insn_t *ir = block->ir_head;
        if (unlikely(!ir->impl(rv, ir, rv->csr_cycle, rv->PC))) {
            /* block should not be extended if execption handler invoked */
            prev = NULL;
            break;
        }
        prev = block;
    }
}

#if RV32_HAS(SYSTEM)
static void trap_handler(riscv_t *rv)
{
    rv_insn_t *ir = mpool_alloc(rv->block_ir_mp);
    assert(ir);


    uint32_t insn;
    while (rv->is_trapped) { /* set to false by sret/mret implementation */
        insn = rv->io.mem_ifetch(rv, rv->PC);
        assert(insn);

        rv_decode(ir, insn);
        ir->impl = dispatch_table[ir->opcode];
        rv->compressed = is_compressed(insn);
        ir->impl(rv, ir, rv->csr_cycle, rv->PC);
    };
}

static bool ppn_is_valid(riscv_t *rv, uint32_t ppn)
{
    vm_attr_t *attr = PRIV(rv);
    const uint32_t nr_pg_max = attr->mem_size / RV_PG_SIZE;
    return ppn < nr_pg_max;
}

#define PAGE_TABLE(ppn)                                               \
    ppn_is_valid(rv, ppn)                                             \
        ? (uint32_t *) (attr->mem->mem_base + (ppn << (RV_PG_SHIFT))) \
        : NULL

/* Walk through page tables and get the corresponding PTE by virtual address if
 * exists
 * @rv: RISC-V emulator
 * @addr: virtual address
 * @return: NULL if a not found or fault else the corresponding PTE
 */
static uint32_t *mmu_walk(riscv_t *rv,
                          const uint32_t addr,
                          uint32_t *level,
                          uint32_t **pte_ref)
{
    vm_attr_t *attr = PRIV(rv);
    uint32_t ppn = rv->csr_satp & MASK(22);
    /* root page table */
    uint32_t *page_table = PAGE_TABLE(ppn);
    if (!page_table)
        return NULL;

    for (int i = 1; i >= 0; i--) {
        *level = 2 - i;
        uint32_t vpn =
            (addr >> RV_PG_SHIFT >> (i * (RV_PG_SHIFT - 2))) & MASK(10);
        uint32_t *pte = page_table + vpn;
        *pte_ref = pte;

        /* PTE XWRV bit in order */
        uint8_t XWRV_bit = (*pte & MASK(4));
        switch (XWRV_bit) {
        case 0b0001: /* next level of the page table */
            ppn = (*pte >> (RV_PG_SHIFT - 2));
            page_table = PAGE_TABLE(ppn);
            if (!page_table)
                return NULL;
            break;
        case 0b0011:
        case 0b0111:
        case 0b1001:
        case 0b1011:
        case 0b1111:
            ppn = (*pte >> (RV_PG_SHIFT - 2));
            if (*level == 1 &&
                unlikely(ppn & MASK(10))) /* misaligned superpage */
                return NULL;
            return pte; /* leaf PTE */
        case 0b0101:
        case 0b1101:
        default:
            return NULL;
        }
    }

    return NULL;
}

/* Verify the PTE and generate corresponding faults if needed
 * @op: the operation
 * @rv: RISC-V emulator
 * @pte: to be verified pte
 * @addr: the corresponding virtual address to cause fault
 * @return: false if a any fault is generated which caused by violating the
 * access permission else true
 */
/* FIXME: handle access fault, addr out of range check */
#define MMU_FAULT_CHECK(op, rv, pte, addr, access_bits) \
    mmu_##op##_fault_check(rv, pte, addr, access_bits)
#define MMU_FAULT_CHECK_IMPL(op, pgfault)                                   \
    static bool mmu_##op##_fault_check(riscv_t *rv, uint32_t *pte,          \
                                       uint32_t addr, uint32_t access_bits) \
    {                                                                       \
        if (!(pte && (*pte & access_bits))) {                               \
            if (satp_cnt >= 2)                                              \
                rv->is_trapped = true;                                      \
            rv_trap_##pgfault(rv, addr);                                    \
            return false;                                                   \
        }                                                                   \
        /* PTE not found, map it in handler */                              \
        if (!pte) {                                                         \
            if (satp_cnt >= 2)                                              \
                rv->is_trapped = true;                                      \
            rv_trap_##pgfault(rv, addr);                                    \
            return false;                                                   \
        }                                                                   \
        /* valid PTE */                                                     \
        return true;                                                        \
    }

MMU_FAULT_CHECK_IMPL(ifetch, pagefault_insn)
MMU_FAULT_CHECK_IMPL(read, pagefault_load)
MMU_FAULT_CHECK_IMPL(write, pagefault_store)

#define get_ppn_and_offset(ppn, offset)                       \
    uint32_t ppn;                                             \
    uint32_t offset;                                          \
    do {                                                      \
        ppn = *pte >> (RV_PG_SHIFT - 2) << RV_PG_SHIFT;       \
        offset = level == 1 ? addr & MASK((RV_PG_SHIFT + 10)) \
                            : addr & MASK(RV_PG_SHIFT);       \
    } while (0)

uint32_t mmu_ifetch(riscv_t *rv, const uint32_t addr)
{
    if (!rv->csr_satp)
        return memory_ifetch(addr);

    uint32_t level;
    uint32_t *pte_ref;
    uint32_t *pte = mmu_walk(rv, addr, &level, &pte_ref);
    bool ok = MMU_FAULT_CHECK(ifetch, rv, pte, addr, PTE_X);

    if (unlikely(!ok)) {
        pte = mmu_walk(rv, addr, &level, &pte_ref);
        pte = pte_ref;
    }

    get_ppn_and_offset(ppn, offset);
    return memory_ifetch(ppn | offset);
}

static void emu_update_uart_interrupts(riscv_t *rv)
{
    vm_attr_t *attr = PRIV(rv);
    u8250_update_interrupts(attr->uart);
    if (attr->uart->pending_ints) {
        attr->plic->active |= IRQ_UART_BIT;
    } else
        attr->plic->active &= ~IRQ_UART_BIT;
    plic_update_interrupts(rv);
}

#define MMIO_PLIC 1
#define MMIO_UART 0
#define MMIO_R 1
#define MMIO_W 0

uint8_t ret_char;
/* clang-format off */
#define MMIO_OP(io, rw)                                           \
    IIF(io)( /* PLIC */                                           \
        IIF(rw)( /* read */                                       \
            read_val = plic_read(rv, (addr & 0x3FFFFFF) >> 2);           \
	    plic_update_interrupts(rv); return read_val;          \
	    ,     /* write */                                     \
            plic_write(rv, (addr & 0x3FFFFFF) >> 2, val);   \
            plic_update_interrupts(rv); return;                   \
        )                                                         \
        ,    /* UART */                                           \
        IIF(rw)( /* read */                                       \
            /*return 0x60 | 0x1; */                                   \
	    ret_char = u8250_read(PRIV(rv)->uart, addr & 0xFFFFF);\
	    emu_update_uart_interrupts(rv);\
	    return ret_char;\
	    ,   /* write */                                       \
	    u8250_write(PRIV(rv)->uart, addr & 0xFFFFF, val);\
	    emu_update_uart_interrupts(rv);\
	    return;\
	)                                                         \
    )
/* clang-format on */

#define MMIO_READ()                                         \
    do {                                                    \
        uint32_t read_val;                                  \
        if ((addr >> 28) == 0xF) { /* MMIO at 0xF_______ */ \
            /* 256 regions of 1MiB */                       \
            switch ((addr >> 20) & MASK(8)) {               \
            case 0x0:                                       \
            case 0x2: /* PLIC (0 - 0x3F) */                 \
                MMIO_OP(MMIO_PLIC, MMIO_R);                 \
            case 0x40: /* UART */                           \
                MMIO_OP(MMIO_UART, MMIO_R);                 \
            }                                               \
        }                                                   \
    } while (0)

#define MMIO_WRITE()                                        \
    do {                                                    \
        if ((addr >> 28) == 0xF) { /* MMIO at 0xF_______ */ \
            /* 256 regions of 1MiB */                       \
            switch ((addr >> 20) & MASK(8)) {               \
            case 0x0:                                       \
            case 0x2: /* PLIC (0 - 0x3F) */                 \
                MMIO_OP(MMIO_PLIC, MMIO_W);                 \
            case 0x40: /* UART */                           \
                MMIO_OP(MMIO_UART, MMIO_W);                 \
            }                                               \
        }                                                   \
    } while (0)

uint32_t mmu_read_w(riscv_t *rv, const uint32_t addr)
{
    if (!rv->csr_satp)
        return memory_read_w(addr);

    uint32_t level;
    uint32_t *pte_ref;
    uint32_t *pte = mmu_walk(rv, addr, &level, &pte_ref);
    bool ok = MMU_FAULT_CHECK(read, rv, pte, addr, PTE_R);
    if (unlikely(!ok)) {
        pte = mmu_walk(rv, addr, &level, &pte_ref);
        pte = pte_ref;
    }

    {
        get_ppn_and_offset(ppn, offset);
        const uint32_t addr = ppn | offset;
        const vm_attr_t *attr = PRIV(rv);
        if (addr < attr->mem->mem_size)
            return memory_read_w(addr);

        MMIO_READ();
    }
}

uint16_t mmu_read_s(riscv_t *rv, const uint32_t addr)
{
    if (!rv->csr_satp)
        return memory_read_s(addr);

    uint32_t level;
    uint32_t *pte_ref;
    uint32_t *pte = mmu_walk(rv, addr, &level, &pte_ref);
    bool ok = MMU_FAULT_CHECK(read, rv, pte, addr, PTE_R);
    if (unlikely(!ok)) {
        pte = mmu_walk(rv, addr, &level, &pte_ref);
        pte = pte_ref;
    }

    {
        get_ppn_and_offset(ppn, offset);
        const uint32_t addr = ppn | offset;
        const vm_attr_t *attr = PRIV(rv);
        if (addr < attr->mem->mem_size) {
            return memory_read_s(addr);
        }
    }
}

uint8_t mmu_read_b(riscv_t *rv, const uint32_t addr)
{
    if (!rv->csr_satp)
        return memory_read_b(addr);

    uint32_t level;
    uint32_t *pte_ref;
    uint32_t *pte = mmu_walk(rv, addr, &level, &pte_ref);
    bool ok = MMU_FAULT_CHECK(read, rv, pte, addr, PTE_R);
    if (unlikely(!ok)) {
        pte = mmu_walk(rv, addr, &level, &pte_ref);
        pte = pte_ref;
    }

    {
        get_ppn_and_offset(ppn, offset);
        const uint32_t addr = ppn | offset;
        const vm_attr_t *attr = PRIV(rv);
        if (addr < attr->mem->mem_size)
            return memory_read_b(addr);

        MMIO_READ();
    }
}

void mmu_write_w(riscv_t *rv, const uint32_t addr, const uint32_t val)
{
    if (!rv->csr_satp)
        return memory_write_w(addr, (uint8_t *) &val);

    uint32_t level;
    uint32_t *pte_ref;
    uint32_t *pte = mmu_walk(rv, addr, &level, &pte_ref);
    bool ok = MMU_FAULT_CHECK(write, rv, pte, addr, PTE_W);
    if (unlikely(!ok)) {
        pte = mmu_walk(rv, addr, &level, &pte_ref);
        pte = pte_ref;
    }

    {
        get_ppn_and_offset(ppn, offset);
        const uint32_t addr = ppn | offset;
        const vm_attr_t *attr = PRIV(rv);
        if (addr < attr->mem->mem_size) {
            memory_write_w(addr, (uint8_t *) &val);
            return;
        }

        MMIO_WRITE();
    }
}

void mmu_write_s(riscv_t *rv, const uint32_t addr, const uint16_t val)
{
    if (!rv->csr_satp)
        return memory_write_s(addr, (uint8_t *) &val);

    uint32_t level;
    uint32_t *pte_ref;
    uint32_t *pte = mmu_walk(rv, addr, &level, &pte_ref);
    bool ok = MMU_FAULT_CHECK(write, rv, pte, addr, PTE_W);
    if (unlikely(!ok)) {
        pte = mmu_walk(rv, addr, &level, &pte_ref);
        pte = pte_ref;
    }

    {
        get_ppn_and_offset(ppn, offset);
        const uint32_t addr = ppn | offset;
        const vm_attr_t *attr = PRIV(rv);
        if (addr < attr->mem->mem_size) {
            memory_write_s(addr, (uint8_t *) &val);
            return;
        }
    }
}

void mmu_write_b(riscv_t *rv, const uint32_t addr, const uint8_t val)
{
    if (!rv->csr_satp)
        return memory_write_b(addr, (uint8_t *) &val);

    uint32_t level;
    uint32_t *pte_ref;
    uint32_t *pte = mmu_walk(rv, addr, &level, &pte_ref);
    bool ok = MMU_FAULT_CHECK(write, rv, pte, addr, PTE_W);
    if (unlikely(!ok)) {
        pte = mmu_walk(rv, addr, &level, &pte_ref);
        pte = pte_ref;
    }

    {
        get_ppn_and_offset(ppn, offset);
        const uint32_t addr = ppn | offset;
        const vm_attr_t *attr = PRIV(rv);
        if (addr < attr->mem->mem_size) {
            memory_write_b(addr, (uint8_t *) &val);
            return;
        }

        MMIO_WRITE();
    }
}

riscv_io_t mmu_io = {
    /* memory read interface */
    .mem_ifetch = mmu_ifetch,
    .mem_read_w = mmu_read_w,
    .mem_read_s = mmu_read_s,
    .mem_read_b = mmu_read_b,

    /* memory write interface */
    .mem_write_w = mmu_write_w,
    .mem_write_s = mmu_write_s,
    .mem_write_b = mmu_write_b,

    /* system services or essential routines */
    .on_ecall = ecall_handler,
    .on_ebreak = ebreak_handler,
    .on_memcpy = memcpy_handler,
    .on_memset = memset_handler,
};
#endif /* SYSTEM */

void ebreak_handler(riscv_t *rv)
{
    assert(rv);
    rv_trap_breakpoint(rv, rv->PC);
}

void ecall_handler(riscv_t *rv)
{
    assert(rv);
#if RV32_HAS(SYSTEM)
    if (rv->priv_mode == RV_PRIV_U_MODE) {
        rv_trap_ecall_U(rv, 0);
    } else if (rv->priv_mode ==
               RV_PRIV_S_MODE) { /* trap to SBI syscall handler */
        rv->PC += 4;
        syscall_handler(rv);
    } else {
        printf("cannot handle ecall here, priv mode: %d\n", rv->priv_mode);
    }
#else
    rv_trap_ecall_M(rv, 0);
    syscall_handler(rv);
#endif
}

void memset_handler(riscv_t *rv)
{
    memory_t *m = PRIV(rv)->mem;
    memset((char *) m->mem_base + rv->X[rv_reg_a0], rv->X[rv_reg_a1],
           rv->X[rv_reg_a2]);
    rv->PC = rv->X[rv_reg_ra] & ~1U;
}

void memcpy_handler(riscv_t *rv)
{
    memory_t *m = PRIV(rv)->mem;
    memcpy((char *) m->mem_base + rv->X[rv_reg_a0],
           (char *) m->mem_base + rv->X[rv_reg_a1], rv->X[rv_reg_a2]);
    rv->PC = rv->X[rv_reg_ra] & ~1U;
}

void dump_registers(riscv_t *rv, char *out_file_path)
{
    FILE *f = out_file_path[0] == '-' ? stdout : fopen(out_file_path, "w");
    if (!f) {
        fprintf(stderr, "Cannot open registers output file.\n");
        return;
    }

    fprintf(f, "{\n");
    for (unsigned i = 0; i < N_RV_REGS; i++) {
        char *comma = i < N_RV_REGS - 1 ? "," : "";
        fprintf(f, "  \"x%d\": %u%s\n", i, rv->X[i], comma);
    }
    fprintf(f, "}\n");

    if (out_file_path[0] != '-')
        fclose(f);
}
