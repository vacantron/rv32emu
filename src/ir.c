#include <assert.h>
#include <stdio.h>

#include <ir.h>
#include <ir_builder.h>

#include "riscv_private.h"

#define MAX_IR_MAP_SIZE 256
#define MAX_IR_STACK_SIZE 2048
#define MAX_IR_LOOP_SUCC 32

static ir_ref ir_reg_refs[N_RV_REGS];

static ir_ref ir_branch_stack[MAX_IR_STACK_SIZE];
static int ir_branch_stack_size = 0;

static int proc_cnt = 0;

static struct ir_map_entry {
    uint32_t pc;
    ir_ref ref;
    ir_ref succ[MAX_IR_LOOP_SUCC];
    size_t succ_size;
};

static struct ir_map {
    size_t size;
    struct ir_map_entry entries[MAX_IR_MAP_SIZE];
};

static void ir_map_insert(struct ir_map *map, uint32_t pc, ir_ref ref)
{
    struct ir_map_entry entry = {.pc = pc, .ref = ref, .succ_size = 0};
    map->entries[map->size++] = entry;
    assert(map->size < MAX_IR_MAP_SIZE);
}

static struct ir_map_entry *ir_map_search(struct ir_map *map, uint32_t pc)
{
    for (size_t i = 0; i < map->size; i++) {
        if (map->entries[i].pc == pc)
            return &map->entries[i];
    }
    return NULL;
}

static inline void ir_reg_refs_clear()
{
    memset((void *) ir_reg_refs, 0, sizeof(ir_ref) * N_RV_REGS);
}

static inline void ir_reg_refs_store_all(ir_ctx *ctx, riscv_t *rv)
{
    for (size_t i = 0; i < N_RV_REGS; i++) {
        if (!ir_reg_refs[i])
            continue;

        ir_ref addr = ir_CONST_ADDR(&rv->X[i]);
        ir_STORE(addr, ir_reg_refs[i]);
    }
}

static inline void ir_bt_set_init(block_t *block)
{
    block->bt_set = calloc(1, sizeof(set_t));
}

static inline bool ir_is_terminal_insn(uint8_t opcode)
{
    switch (opcode) {
    case rv_insn_jalr:
    case rv_insn_ecall:
    case rv_insn_ebreak:
    case rv_insn_sret:
    case rv_insn_mret:
#if RV32_HAS(EXT_C)
    case rv_insn_cjalr:
    case rv_insn_cjr:
    case rv_insn_cebreak:
#endif
        return true;
    default:
        return false;
    }
    __UNREACHABLE;
}

static void ir_find_loop(riscv_t *rv, block_t *block)
{
    if (!set_add((set_t *) block->bt_set, block->ir_head->pc))
        assert(!"Should not touch existent block again");

    block->has_ir_loop = false;

    rv_insn_t *tail = block->ir_tail;

    if (ir_is_terminal_insn(tail->opcode))
        return;

    if (tail->branch_taken) {
        block_t *next_block =
            cache_get(rv->block_cache, tail->branch_taken->pc, false);

        if (set_has((set_t *) block->bt_set, tail->branch_taken->pc)) {
            next_block->has_ir_loop = true;
            next_block->proc_cnt = proc_cnt;
        } else {
            if (next_block->proc_cnt != proc_cnt) {
                next_block->proc_cnt = proc_cnt;
                next_block->has_ir_loop = false;
            }
            if (!next_block->has_ir_loop) {
                if (!next_block->bt_set)
                    ir_bt_set_init(next_block);

                memcpy(next_block->bt_set, block->bt_set, sizeof(set_t));
                ir_find_loop(rv, next_block);
            }
        }
    }
    if (tail->branch_untaken) {
        block_t *next_block =
            cache_get(rv->block_cache, tail->branch_untaken->pc, false);

        if (set_has((set_t *) block->bt_set, tail->branch_untaken->pc)) {
            next_block->has_ir_loop = true;
            next_block->proc_cnt = proc_cnt;
        } else {
            if (next_block->proc_cnt != proc_cnt) {
                next_block->proc_cnt = proc_cnt;
                next_block->has_ir_loop = false;
            }
            if (!next_block->has_ir_loop) {
                if (!next_block->bt_set)
                    ir_bt_set_init(next_block);

                memcpy(next_block->bt_set, block->bt_set, sizeof(set_t));
                ir_find_loop(rv, next_block);
            }
        }
    }
}

static inline void ir_prepare_operand(ir_ctx *ctx, riscv_t *rv, uint8_t idx)
{
    if (ir_reg_refs[idx])
        return;

    ir_reg_refs[idx] =
        ir_HARD_COPY_I32(ir_LOAD_I32(ir_CONST_ADDR(&rv->X[idx])));
}

void ir_build(ir_ctx *ctx,
              riscv_t *rv,
              block_t *block,
              set_t *set,
              struct ir_map *map)
{
    rv_insn_t *curr = block->ir_head;
    uintptr_t mem_base = (uintptr_t) PRIV(rv)->mem->mem_base;

    if (set_has(set, curr->pc))
        return;

    if (block->has_ir_loop) {
        set_add(set, curr->pc);
        ir_map_insert(map, curr->pc, ir_LOOP_BEGIN(ir_END()));
    }

    ir_reg_refs_clear();

    while (1) {
        ir_ref addr, val, cond;

        switch (curr->opcode) {
        case rv_insn_nop:
            break;
        case rv_insn_auipc:
            addr = ir_CONST_ADDR(&rv->PC);
            ir_reg_refs[curr->rd] =
                ir_ADD_I32(ir_CONST_I32(curr->imm), ir_LOAD_I32(addr));
            break;
        case rv_insn_lui:
            ir_reg_refs[curr->rd] = ir_CONST_I32(curr->imm);
            break;
        case rv_insn_add:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_ADD_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_sub:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_SUB_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_xor:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_XOR_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_or:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_OR_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_and:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_AND_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_sll:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] = ir_SHL_I32(
                ir_reg_refs[curr->rs1],
                ir_AND_I32(ir_reg_refs[curr->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_srl:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] = ir_SHR_I32(
                ir_reg_refs[curr->rs1],
                ir_AND_I32(ir_reg_refs[curr->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_sra:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] = ir_SAR_I32(
                ir_reg_refs[curr->rs1],
                ir_AND_I32(ir_reg_refs[curr->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_addi:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rd] = ir_ADD_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_andi:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rd] = ir_AND_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_ori:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rd] = ir_OR_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_xori:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rd] = ir_XOR_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_slli:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->imm & 0x1f);
            ir_reg_refs[curr->rd] = ir_SHL_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_srli:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->imm & 0x1f);
            ir_reg_refs[curr->rd] = ir_SHR_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_srai:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->imm & 0x1f);
            ir_reg_refs[curr->rd] = ir_SAR_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_lw:
            ir_prepare_operand(ctx, rv, curr->rs1);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(addr, ir_reg_refs[curr->rs1]);
            /* FIXME: Do NOT remove hard copy. IR does not truncate loaded value
             * when fusing operators
             */
            ir_reg_refs[curr->rd] = ir_HARD_COPY_I32(ir_LOAD_I32(addr));
            break;
        case rv_insn_lh:
            ir_prepare_operand(ctx, rv, curr->rs1);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(addr, ir_reg_refs[curr->rs1]);
            ir_reg_refs[curr->rd] = ir_SEXT_I32(ir_LOAD_I16(addr));
            break;
        case rv_insn_lb:
            ir_prepare_operand(ctx, rv, curr->rs1);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(addr, ir_reg_refs[curr->rs1]);
            ir_reg_refs[curr->rd] = ir_SEXT_I32(ir_LOAD_I8(addr));
            break;
        case rv_insn_lhu:
            ir_prepare_operand(ctx, rv, curr->rs1);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(addr, ir_reg_refs[curr->rs1]);
            ir_reg_refs[curr->rd] = ir_ZEXT_I32(ir_LOAD_U16(addr));
            break;
        case rv_insn_lbu:
            ir_prepare_operand(ctx, rv, curr->rs1);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(addr, ir_reg_refs[curr->rs1]);
            ir_reg_refs[curr->rd] = ir_ZEXT_I32(ir_LOAD_U8(addr));
            break;
        case rv_insn_sw:
            ir_prepare_operand(ctx, rv, curr->rs1);

            if (!ir_reg_refs[curr->rs2]) {
                addr = ir_CONST_ADDR(&rv->X[curr->rs2]);
                ir_reg_refs[curr->rs2] = ir_LOAD_I32(addr);
            }

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(ir_reg_refs[curr->rs1], addr);
            ir_STORE(addr, ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_sh:
            ir_prepare_operand(ctx, rv, curr->rs1);

            if (!ir_reg_refs[curr->rs2]) {
                addr = ir_CONST_ADDR(&rv->X[curr->rs2]);
                ir_reg_refs[curr->rs2] = ir_LOAD_I16(addr);
            }

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(ir_reg_refs[curr->rs1], addr);
            ir_STORE(addr, ir_reg_refs[curr->rs2]);

            ir_reg_refs[curr->rs2] = 0;
            break;
        case rv_insn_sb:
            ir_prepare_operand(ctx, rv, curr->rs1);

            if (!ir_reg_refs[curr->rs2]) {
                addr = ir_CONST_ADDR(&rv->X[curr->rs2]);
                ir_reg_refs[curr->rs2] = ir_LOAD_I8(addr);
            }

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(ir_reg_refs[curr->rs1], addr);
            ir_STORE(addr, ir_reg_refs[curr->rs2]);

            ir_reg_refs[curr->rs2] = 0;
            break;
        case rv_insn_slt:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] = ir_COND_I32(
                ir_LT(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]),
                ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_sltu:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] = ir_COND_I32(
                ir_ULT(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]),
                ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_slti:
            ir_prepare_operand(ctx, rv, curr->rs1);

            ir_reg_refs[curr->rd] = ir_COND_I32(
                ir_LT(ir_reg_refs[curr->rs1], ir_CONST_I32(curr->imm)),
                ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_sltiu:
            ir_prepare_operand(ctx, rv, curr->rs1);

            ir_reg_refs[curr->rd] = ir_COND_I32(
                ir_ULT(ir_reg_refs[curr->rs1], ir_CONST_I32(curr->imm)),
                ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_beq:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_EQ(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]));
            ir_IF_TRUE(cond);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            ir_branch_stack[ir_branch_stack_size++] = cond;
            assert(ir_branch_stack_size < MAX_IR_STACK_SIZE);
            break;
        case rv_insn_bne:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_NE(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]));
            ir_IF_TRUE(cond);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            ir_branch_stack[ir_branch_stack_size++] = cond;
            assert(ir_branch_stack_size < MAX_IR_STACK_SIZE);
            break;
        case rv_insn_bge:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_GE(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]));
            ir_IF_TRUE(cond);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            ir_branch_stack[ir_branch_stack_size++] = cond;
            assert(ir_branch_stack_size < MAX_IR_STACK_SIZE);
            break;
        case rv_insn_blt:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_LT(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]));
            ir_IF_TRUE(cond);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            ir_branch_stack[ir_branch_stack_size++] = cond;
            assert(ir_branch_stack_size < MAX_IR_STACK_SIZE);
            break;
        case rv_insn_bgeu:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond =
                ir_IF(ir_UGE(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]));
            ir_IF_TRUE(cond);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            ir_branch_stack[ir_branch_stack_size++] = cond;
            assert(ir_branch_stack_size < MAX_IR_STACK_SIZE);
            break;
        case rv_insn_bltu:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond =
                ir_IF(ir_ULT(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]));
            ir_IF_TRUE(cond);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            ir_branch_stack[ir_branch_stack_size++] = cond;
            assert(ir_branch_stack_size < MAX_IR_STACK_SIZE);
            break;
        case rv_insn_jal:
            ir_reg_refs_store_all(ctx, rv);

            if (curr->rd) {
                addr = ir_CONST_ADDR(&rv->X[curr->rd]);
                val = ir_CONST_U32(curr->pc + 4);
                ir_STORE(addr, val);
            }

            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            break;
        case rv_insn_jalr:
            ir_prepare_operand(ctx, rv, curr->rs1);

            ir_reg_refs_store_all(ctx, rv);

            if (curr->rd) {
                addr = ir_CONST_ADDR(&rv->X[curr->rd]);
                val = ir_CONST_U32(curr->pc + 4);
                ir_STORE(addr, val);
            }

            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->imm);
            ir_reg_refs[curr->rs1] = ir_ADD_U32(
                ir_reg_refs[curr->rs1], ir_AND_U32(val, ir_CONST_U32(~1U)));
            ir_STORE(addr, ir_reg_refs[curr->rs1]);
            break;
        case rv_insn_ecall:
            ir_reg_refs_store_all(ctx, rv);
            addr = ir_CONST_ADDR(&rv->PC);
            ir_STORE(addr, ir_CONST_U32(curr->pc));
            addr = ir_CONST_ADDR((uintptr_t) rv->io.on_ecall);
            ir_CALL_1(IR_UNUSED, addr, ir_CONST_ADDR(rv));
            break;
        case rv_insn_ebreak:
            ir_reg_refs_store_all(ctx, rv);
            addr = ir_CONST_ADDR(&rv->PC);
            ir_STORE(addr, ir_CONST_U32(curr->pc));
            addr = ir_CONST_ADDR((uintptr_t) rv->io.on_ebreak);
            ir_CALL_1(IR_UNUSED, addr, ir_CONST_ADDR(rv));
            break;
#if RV32_HAS(EXT_M)
        case rv_insn_mul:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_MUL_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_mulh: {
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_ref t1 = ir_SEXT_I64(ir_reg_refs[curr->rs1]);
            ir_ref t2 = ir_SEXT_I64(ir_reg_refs[curr->rs2]);

            ir_reg_refs[curr->rd] =
                ir_SHR_I64(ir_MUL_I64(t1, t2), ir_CONST_I32(32));
        } break;
        case rv_insn_mulhsu: {
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_ref t1 = ir_SEXT_I64(ir_reg_refs[curr->rs1]);
            ir_ref t2 = ir_ZEXT_I64(ir_reg_refs[curr->rs2]);

            ir_reg_refs[curr->rd] =
                ir_SHR_I64(ir_MUL_I64(t1, t2), ir_CONST_I32(32));
        } break;
        case rv_insn_mulhu: {
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_ref t1 = ir_ZEXT_I64(ir_reg_refs[curr->rs1]);
            ir_ref t2 = ir_ZEXT_I64(ir_reg_refs[curr->rs2]);

            ir_reg_refs[curr->rd] =
                ir_SHR_U64(ir_MUL_U64(t1, t2), ir_CONST_U32(32));
        } break;
        case rv_insn_div: {
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_DIV_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
        } break;
        case rv_insn_divu: {
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_DIV_U32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
        } break;
        case rv_insn_rem: {
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_MOD_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
        } break;
        case rv_insn_remu: {
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_MOD_U32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
        } break;
#endif
#if RV32_HAS(EXT_C)
        case rv_insn_cnop:
            break;
        case rv_insn_cli:
        case rv_insn_clui:
            ir_reg_refs[curr->rd] = ir_CONST_I32(curr->imm);
            break;
        case rv_insn_cmv:
            ir_prepare_operand(ctx, rv, curr->rs2);
            ir_reg_refs[curr->rd] =
                ir_ADD_I32(ir_reg_refs[curr->rs2], ir_CONST_I32(0));
            break;
        case rv_insn_cadd:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_ADD_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_csub:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_SUB_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_cand:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_AND_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_cor:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_OR_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_cxor:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            ir_reg_refs[curr->rd] =
                ir_XOR_I32(ir_reg_refs[curr->rs1], ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_cslli:
            ir_prepare_operand(ctx, rv, curr->rd);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rd] = ir_SHL_I32(ir_reg_refs[curr->rd], val);
            break;
        case rv_insn_csrli:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->shamt);
            ir_reg_refs[curr->rs1] = ir_SHR_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_csrai:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->shamt);
            ir_reg_refs[curr->rs1] = ir_SAR_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_caddi:
            ir_prepare_operand(ctx, rv, curr->rd);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rd] = ir_ADD_I32(ir_reg_refs[curr->rd], val);
            break;
        case rv_insn_candi:
            ir_prepare_operand(ctx, rv, curr->rs1);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rs1] = ir_AND_I32(ir_reg_refs[curr->rs1], val);
            break;
        case rv_insn_caddi4spn:
            ir_prepare_operand(ctx, rv, rv_reg_sp);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rd] = ir_ADD_I32(ir_reg_refs[rv_reg_sp], val);
            break;
        case rv_insn_caddi16sp:
            ir_prepare_operand(ctx, rv, curr->rd);

            val = ir_CONST_I32(curr->imm);
            ir_reg_refs[curr->rd] = ir_ADD_I32(ir_reg_refs[curr->rd], val);
            break;
        case rv_insn_clwsp:
            ir_prepare_operand(ctx, rv, rv_reg_sp);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(addr, ir_reg_refs[rv_reg_sp]);
            ir_reg_refs[curr->rd] = ir_HARD_COPY_I32(ir_LOAD_I32(addr));
            break;
        case rv_insn_cswsp:
            ir_prepare_operand(ctx, rv, rv_reg_sp);
            ir_prepare_operand(ctx, rv, curr->rs2);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(ir_reg_refs[rv_reg_sp], addr);
            ir_STORE(addr, ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_clw:
            ir_prepare_operand(ctx, rv, curr->rs1);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(addr, ir_reg_refs[curr->rs1]);
            ir_reg_refs[curr->rd] = ir_HARD_COPY_I32(ir_LOAD_I32(addr));
            break;
        case rv_insn_csw:
            ir_prepare_operand(ctx, rv, curr->rs1);
            ir_prepare_operand(ctx, rv, curr->rs2);

            addr = ir_CONST_ADDR(mem_base + curr->imm);
            addr = ir_ADD_A(ir_reg_refs[curr->rs1], addr);
            ir_STORE(addr, ir_reg_refs[curr->rs2]);
            break;
        case rv_insn_cbeqz:
            ir_prepare_operand(ctx, rv, curr->rs1);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_EQ(ir_reg_refs[curr->rs1], ir_CONST_I32(0)));
            ir_IF_TRUE(cond);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            ir_branch_stack[ir_branch_stack_size++] = cond;
            assert(ir_branch_stack_size < MAX_IR_STACK_SIZE);
            break;
        case rv_insn_cbnez:
            ir_prepare_operand(ctx, rv, curr->rs1);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_NE(ir_reg_refs[curr->rs1], ir_CONST_I32(0)));
            ir_IF_TRUE(cond);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            ir_branch_stack[ir_branch_stack_size++] = cond;
            assert(ir_branch_stack_size < MAX_IR_STACK_SIZE);
            break;
        case rv_insn_cj:
            ir_reg_refs_store_all(ctx, rv);
            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            break;
        case rv_insn_cjal:
            ir_reg_refs_store_all(ctx, rv);

            addr = ir_CONST_ADDR(&rv->X[rv_reg_ra]);
            val = ir_CONST_U32(curr->pc + 2);
            ir_STORE(addr, val);

            addr = ir_CONST_ADDR(&rv->PC);
            val = ir_CONST_U32(curr->pc + curr->imm);
            ir_STORE(addr, val);
            break;
        case rv_insn_cjr:
            ir_prepare_operand(ctx, rv, curr->rs1);

            ir_reg_refs_store_all(ctx, rv);

            addr = ir_CONST_ADDR(&rv->PC);
            ir_STORE(addr, ir_reg_refs[curr->rs1]);
            break;
        case rv_insn_cjalr:
            ir_prepare_operand(ctx, rv, curr->rs1);

            ir_reg_refs_store_all(ctx, rv);

            addr = ir_CONST_ADDR(&rv->X[rv_reg_ra]);
            val = ir_CONST_U32(curr->pc + 2);
            ir_STORE(addr, val);

            addr = ir_CONST_ADDR(&rv->PC);
            ir_STORE(addr, ir_reg_refs[curr->rs1]);
            break;
        case rv_insn_cebreak:
            ir_reg_refs_store_all(ctx, rv);
            addr = ir_CONST_ADDR(&rv->PC);
            ir_STORE(addr, ir_CONST_U32(curr->pc));
            addr = ir_CONST_ADDR((uintptr_t) rv->io.on_ebreak);
            ir_CALL_1(IR_UNUSED, addr, ir_CONST_ADDR(rv));
            break;
#endif
        default:
            assert(!"Unsupported operator");
            __UNREACHABLE;
        }

        if (!curr->next)
            break;

        curr = curr->next;
    }

    if (ir_is_terminal_insn(curr->opcode)) {
        ir_RETURN(IR_UNUSED);
        return;
    }

    if (curr->branch_taken) {
        if (set_has(set, curr->branch_taken->pc)) {
            struct ir_map_entry *taken =
                ir_map_search(map, curr->branch_taken->pc);

            taken->succ[taken->succ_size++] = ir_END();
            assert(taken->succ_size < MAX_IR_LOOP_SUCC);
        } else {
            block_t *next_block =
                cache_get(rv->block_cache, curr->branch_taken->pc, false);
            ir_build(ctx, rv, next_block, set, map);
        }

        if (curr->opcode != rv_insn_jal && curr->opcode != rv_insn_cj &&
            curr->opcode != rv_insn_cjal) {
            ir_IF_FALSE(ir_branch_stack[--ir_branch_stack_size]);
            assert(ir_branch_stack_size > -1);
        }
    } else {
        ir_RETURN(IR_UNUSED);
        if (curr->opcode != rv_insn_jal && curr->opcode != rv_insn_cj &&
            curr->opcode != rv_insn_cjal) {
            ir_IF_FALSE(ir_branch_stack[--ir_branch_stack_size]);
            assert(ir_branch_stack_size > -1);
        }
    }

    if (curr->opcode != rv_insn_jal && curr->opcode != rv_insn_cj &&
        curr->opcode != rv_insn_cjal) {
        if (curr->opcode == rv_insn_cbeqz || curr->opcode == rv_insn_cbnez)
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_U32(curr->pc + 2));
        else
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_U32(curr->pc + 4));
    }

    if (curr->branch_untaken) {
        if (set_has(set, curr->branch_untaken->pc)) {
            struct ir_map_entry *untaken =
                ir_map_search(map, curr->branch_untaken->pc);
            untaken->succ[untaken->succ_size++] = ir_END();
            assert(untaken->succ_size < MAX_IR_LOOP_SUCC);
        } else {
            block_t *next_block =
                cache_get(rv->block_cache, curr->branch_untaken->pc, false);
            ir_build(ctx, rv, next_block, set, map);
        }
    } else {
        if (curr->opcode != rv_insn_jal && curr->opcode != rv_insn_cj &&
            curr->opcode != rv_insn_cjal)
            ir_RETURN(IR_UNUSED);
    }
}

void ir_compile(riscv_t *rv, block_t *block)
{
    set_t set;
    struct ir_map map;

    proc_cnt++;

    set_reset(&set);
    memset((void *) &map, 0, sizeof(struct ir_map));

    ir_ctx *ctx = malloc(sizeof(ir_ctx));

    ir_init(ctx,
            IR_FUNCTION | IR_OPT_INLINE | IR_OPT_FOLDING | IR_OPT_CFG |
                IR_OPT_CODEGEN,
            256, 1024);
    ctx->ret_type = IR_VOID;

    ir_bt_set_init(block);
    block->proc_cnt = proc_cnt;
    ir_find_loop(rv, block);

    ir_consistency_check();

    ir_START();
    ir_build(ctx, rv, block, &set, &map);

    for (size_t i = 0; i < map.size; i++) {
        assert(map.entries[i].succ_size > 0);

        ir_ref refs[MAX_IR_LOOP_SUCC];
        for (size_t j = 0; j < map.entries[i].succ_size; j++)
            refs[j] = map.entries[i].succ[j];

        ir_MERGE_N(map.entries[i].succ_size, refs);
        ir_MERGE_SET_OP(map.entries[i].ref, 2, ir_LOOP_END());
    }

    ir_consistency_check();

    size_t size;
    block->ir_func_ptr = ir_jit_compile(ctx, 2, &size);
    block->ir_hot = true;

    ir_free(ctx);
    free(ctx);

    /* TODO: free "bt_set" */
}
