/*
 * rv32emu is freely redistributable under the MIT License. See the file
 * "LICENSE" for information on usage and redistribution of this file.
 */

#include <assert.h>

#include <ir.h>
#include <ir_builder.h>

#include "riscv_private.h"

#define MAX_T2C_MAP_SIZE (1 << 8)
#define MAX_T2C_STACK_SIZE (1 << 8)
#define MAX_T2C_VISITED_BLOCKS (1 << 8)
#define MAX_T2C_LOOP_TERMINATOR (1 << 8)

static uint32_t proc_cnt = 0;

/* lazy storing of vm registers */
static ir_ref t2c_vregs[N_RV_REGS];

static ir_ref t2c_branches[MAX_T2C_STACK_SIZE];
static size_t t2c_branches_idx = 0;

struct ir_map_entry {
    uint32_t pc;
    ir_ref head;
    ir_ref ends[MAX_T2C_LOOP_TERMINATOR];
    size_t ends_idx;
};

static void t2c_branch_push(ir_ref e)
{
    t2c_branches[t2c_branches_idx++] = e;
    assert(t2c_branches_idx <= MAX_T2C_STACK_SIZE);
}

static ir_ref t2c_branch_pop()
{
    assert(t2c_branches_idx > 0);
    return t2c_branches[--t2c_branches_idx];
}

static void t2c_branch_connect(struct ir_map_entry *entry, ir_ref e)
{
    entry->ends[entry->ends_idx++] = e;
    assert(entry->ends_idx <= MAX_T2C_LOOP_TERMINATOR);
}

struct ir_map {
    size_t size;
    struct ir_map_entry entries[MAX_T2C_MAP_SIZE];
};

static void t2c_map_insert(struct ir_map *map, uint32_t pc, ir_ref head)
{
    struct ir_map_entry entry = {.pc = pc, .head = head, .ends_idx = 0};
    map->entries[map->size++] = entry;

    assert(map->size <= MAX_T2C_MAP_SIZE);
}

static struct ir_map_entry *t2c_map_search(struct ir_map *map, uint32_t pc)
{
    for (size_t i = 0; i < map->size; i++) {
        if (map->entries[i].pc == pc)
            return &map->entries[i];
    }
    return NULL;
}

static inline void t2c_clear_vregs()
{
    memset((void *) t2c_vregs, 0, sizeof(ir_ref) * N_RV_REGS);
}

static inline void t2c_store_all_vregs(ir_ctx *ctx, riscv_t *rv)
{
    for (size_t i = 0; i < N_RV_REGS; i++) {
        if (!t2c_vregs[i])
            continue;

        ir_STORE(ir_CONST_ADDR(&rv->X[i]),
                 ir_TRUNC_I32(ir_ZEXT_I64(t2c_vregs[i])));
    }
}

static inline bool t2c_is_indirect_branch(uint8_t opcode)
{
    switch (opcode) {
    case rv_insn_jalr:
    case rv_insn_ecall:
    case rv_insn_ebreak:
#if RV32_HAS(SYSTEM)
    case rv_insn_sret:
#endif
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


static block_t *t2c_visited_bb[MAX_T2C_VISITED_BLOCKS];
static int t2c_visited_bb_cnt = 0;

static void t2c_visited_bb_push(block_t *block)
{
    t2c_visited_bb[t2c_visited_bb_cnt++] = block;
    assert(t2c_visited_bb_cnt <= MAX_T2C_VISITED_BLOCKS);
}

static block_t *t2c_visited_bb_pop()
{
    assert(t2c_visited_bb_cnt > 0);
    return t2c_visited_bb[--t2c_visited_bb_cnt];
}

static block_t *t2c_visited_bb_peek()
{
    return t2c_visited_bb[t2c_visited_bb_cnt - 1];
}

static bool t2c_is_bb_visited(block_t *block)
{
    for (int i = 0; i < t2c_visited_bb_cnt; i++) {
        if (t2c_visited_bb[i] == block)
            return true;
    }
    return false;
}

static void t2c_detect_loop(riscv_t *rv)
{
    block_t *block = t2c_visited_bb_peek();
    rv_insn_t *tail = block->ir_tail;

    if (t2c_is_indirect_branch(tail->opcode))
        return;

    if (tail->branch_taken) {
        block_t *next =
            cache_get(rv->block_cache, tail->branch_taken->pc, false);
        if (next->proc_cnt != proc_cnt) {
            next->proc_cnt = proc_cnt;
            next->t2c_has_loop = false;
            next->t2_need_merge = false;
            next->t2c_visited_by = block;
        } else {
            if (next->t2c_visited_by != block) {
                next->t2_need_merge = true;
            }
        }

        if (!t2c_is_bb_visited(next)) {
            t2c_visited_bb_push(next);
            t2c_detect_loop(rv);
            t2c_visited_bb_pop();
        } else {
            next->t2c_has_loop = true;
        }
    }

    if (tail->branch_untaken) {
        block_t *next =
            cache_get(rv->block_cache, tail->branch_untaken->pc, false);
        if (next->proc_cnt != proc_cnt) {
            next->proc_cnt = proc_cnt;
            next->t2c_has_loop = false;
            next->t2_need_merge = false;
            next->t2c_visited_by = block;
        } else {
            if (next->t2c_visited_by != block) {
                next->t2_need_merge = true;
            }
        }

        if (!t2c_is_bb_visited(next)) {
            t2c_visited_bb_push(next);
            t2c_detect_loop(rv);
            t2c_visited_bb_pop();
        } else {
            next->t2c_has_loop = true;
        }
    }
}

static inline void t2c_prepare_operand(ir_ctx *ctx, riscv_t *rv, uint8_t idx)
{
    assert(idx < 32);

    if (t2c_vregs[idx]) {
        t2c_vregs[idx] = ir_TRUNC_I32(ir_ZEXT_I64(t2c_vregs[idx]));
        return;
    }

    t2c_vregs[idx] = ir_LOAD_I32(ir_CONST_ADDR(&rv->X[idx]));
}

void t2c_build(ir_ctx *ctx,
               riscv_t *rv,
               block_t *block,
               set_t *set,
               struct ir_map *loop_map,
               struct ir_map *merge_map)
{
    uintptr_t mem_base = (uintptr_t) PRIV(rv)->mem->mem_base;
    rv_insn_t *ir = block->ir_head;

    if (set_has(set, ir->pc))
        return;

    set_add(set, ir->pc);

    if (block->proc_cnt == proc_cnt) {
        /*
         * Use the branch "dev" of jit-framework to create dangling label.
         */
        if (block->t2c_has_loop) {
            t2c_map_insert(loop_map, ir->pc, ir_LOOP_BEGIN(ir_END()));
        } else if (block->t2_need_merge) {
            t2c_map_insert(merge_map, ir->pc, ir_LOOP_BEGIN(ir_END()));
        }
    }

    t2c_clear_vregs();

    while (1) {
        ir_ref cond;

        switch (ir->opcode) {
        case rv_insn_nop:
            break;
        case rv_insn_auipc:
            /* incorrect immediate value has been fixed */
            t2c_vregs[ir->rd] = ir_CONST_I32(ir->imm + ir->pc);
            break;
        case rv_insn_lui:
            t2c_vregs[ir->rd] = ir_CONST_I32(ir->imm);
            break;
        case rv_insn_add:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_ADD_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_sub:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_SUB_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_xor:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_XOR_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_or:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_OR_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_and:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_AND_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_sll:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_SHL_I32(t2c_vregs[ir->rs1],
                           ir_AND_I32(t2c_vregs[ir->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_srl:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_SHR_I32(t2c_vregs[ir->rs1],
                           ir_AND_I32(t2c_vregs[ir->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_sra:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_SAR_I32(t2c_vregs[ir->rs1],
                           ir_AND_I32(t2c_vregs[ir->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_addi:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] =
                ir_ADD_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm));
            break;
        case rv_insn_andi:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] =
                ir_AND_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm));
            break;
        case rv_insn_ori:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] =
                ir_OR_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm));
            break;
        case rv_insn_xori:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] =
                ir_XOR_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm));
            break;
        case rv_insn_slli:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] =
                ir_SHL_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm & 0x1f));
            break;
        case rv_insn_srli:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] =
                ir_SHR_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm & 0x1f));
            break;
        case rv_insn_srai:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] =
                ir_SAR_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm & 0x1f));
            break;
        case rv_insn_lw:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] = ir_LOAD_I32(ir_ADD_OFFSET(
                ir_ZEXT_I64(t2c_vregs[ir->rs1]), mem_base + ir->imm));
            break;
        case rv_insn_lh:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] = ir_SEXT_I32(ir_LOAD_I16(ir_ADD_OFFSET(
                ir_ZEXT_I64(t2c_vregs[ir->rs1]), mem_base + ir->imm)));
            break;
        case rv_insn_lb:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] = ir_SEXT_I32(ir_LOAD_I8(ir_ADD_OFFSET(
                ir_ZEXT_I64(t2c_vregs[ir->rs1]), mem_base + ir->imm)));
            break;
        case rv_insn_lhu:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] = ir_ZEXT_I32(ir_LOAD_I16(ir_ADD_OFFSET(
                ir_ZEXT_I64(t2c_vregs[ir->rs1]), mem_base + ir->imm)));
            break;
        case rv_insn_lbu:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] = ir_ZEXT_I32(ir_LOAD_I8(ir_ADD_OFFSET(
                ir_ZEXT_I64(t2c_vregs[ir->rs1]), mem_base + ir->imm)));
            break;
        case rv_insn_sw:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            if (!t2c_vregs[ir->rs2]) {
                t2c_vregs[ir->rs2] =
                    ir_LOAD_I32(ir_CONST_ADDR(&rv->X[ir->rs2]));
            } else {
                t2c_vregs[ir->rs2] =
                    ir_TRUNC_I32(ir_ZEXT_I64(t2c_vregs[ir->rs2]));
            }

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(t2c_vregs[ir->rs1]),
                                   mem_base + ir->imm),
                     t2c_vregs[ir->rs2]);
            break;
        case rv_insn_sh:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            if (!t2c_vregs[ir->rs2]) {
                t2c_vregs[ir->rs2] =
                    ir_LOAD_I16(ir_CONST_ADDR(&rv->X[ir->rs2]));
            } else {
                ir_STORE(ir_CONST_ADDR(&rv->X[ir->rs2]), t2c_vregs[ir->rs2]);
                t2c_vregs[ir->rs2] = ir_TRUNC_I16(t2c_vregs[ir->rs2]);
            }

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(t2c_vregs[ir->rs1]),
                                   mem_base + ir->imm),
                     t2c_vregs[ir->rs2]);

            /* drop 16-bit value */
            t2c_vregs[ir->rs2] = 0;
            break;
        case rv_insn_sb:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            if (!t2c_vregs[ir->rs2]) {
                t2c_vregs[ir->rs2] = ir_LOAD_I8(ir_CONST_ADDR(&rv->X[ir->rs2]));
            } else {
                ir_STORE(ir_CONST_ADDR(&rv->X[ir->rs2]), t2c_vregs[ir->rs2]);
                t2c_vregs[ir->rs2] = ir_TRUNC_I8(t2c_vregs[ir->rs2]);
            }

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(t2c_vregs[ir->rs1]),
                                   mem_base + ir->imm),
                     t2c_vregs[ir->rs2]);

            /* drop 8-bit value */
            t2c_vregs[ir->rs2] = 0;
            break;
        case rv_insn_slt:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] = ir_COND_I32(
                ir_ZEXT_I32(ir_LT(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2])),
                ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_sltu:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] = ir_COND_I32(
                ir_ZEXT_I32(ir_ULT(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2])),
                ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_slti:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] = ir_COND_I32(
                ir_ZEXT_I32(ir_LT(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm))),
                ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_sltiu:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] = ir_COND_I32(
                ir_ZEXT_I32(ir_ULT(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm))),
                ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_beq:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_store_all_vregs(ctx, rv);

            cond = ir_IF(ir_EQ(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]));
            t2c_branch_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_bne:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_store_all_vregs(ctx, rv);

            cond = ir_IF(ir_NE(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]));
            t2c_branch_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_bge:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_store_all_vregs(ctx, rv);

            cond = ir_IF(ir_GE(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]));
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            t2c_branch_push(cond);
            break;
        case rv_insn_blt:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_store_all_vregs(ctx, rv);

            cond = ir_IF(ir_LT(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]));
            t2c_branch_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_bgeu:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_store_all_vregs(ctx, rv);

            cond = ir_IF(ir_UGE(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]));
            t2c_branch_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_bltu:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_store_all_vregs(ctx, rv);

            cond = ir_IF(ir_ULT(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]));
            t2c_branch_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_jal:
            t2c_store_all_vregs(ctx, rv);

            if (ir->rd) {
                ir_STORE(ir_CONST_ADDR(&rv->X[ir->rd]),
                         ir_CONST_I32(ir->pc + 4));
            }

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_jalr:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_store_all_vregs(ctx, rv);

            if (ir->rd) {
                ir_STORE(ir_CONST_ADDR(&rv->X[ir->rd]),
                         ir_CONST_I32(ir->pc + 4));
            }

            ir_STORE(
                ir_CONST_ADDR(&rv->PC),
                ir_ADD_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm & ~1U)));
            break;
        case rv_insn_ecall:
            t2c_store_all_vregs(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc));
            ir_CALL_1(IR_UNUSED, ir_CONST_ADDR((uintptr_t) rv->io.on_ecall),
                      ir_CONST_ADDR(rv));
            break;
        case rv_insn_ebreak:
            t2c_store_all_vregs(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc));
            ir_CALL_1(IR_UNUSED, ir_CONST_ADDR((uintptr_t) rv->io.on_ebreak),
                      ir_CONST_ADDR(rv));
            break;
#if RV32_HAS(EXT_M)
        case rv_insn_mul:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_MUL_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_mulh: {
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            ir_ref t1 = ir_SEXT_I64(t2c_vregs[ir->rs1]);
            ir_ref t2 = ir_SEXT_I64(t2c_vregs[ir->rs2]);

            t2c_vregs[ir->rd] =
                ir_TRUNC_I32(ir_SHR_I64(ir_MUL_I64(t1, t2), ir_CONST_I64(32)));
        } break;
        case rv_insn_mulhsu: {
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            ir_ref t1 = ir_SEXT_I64(t2c_vregs[ir->rs1]);
            ir_ref t2 = ir_ZEXT_I64(t2c_vregs[ir->rs2]);

            t2c_vregs[ir->rd] =
                ir_TRUNC_I32(ir_SHR_I64(ir_MUL_I64(t1, t2), ir_CONST_I64(32)));
        } break;
        case rv_insn_mulhu: {
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            ir_ref t1 = ir_ZEXT_I64(t2c_vregs[ir->rs1]);
            ir_ref t2 = ir_ZEXT_I64(t2c_vregs[ir->rs2]);

            t2c_vregs[ir->rd] =
                ir_TRUNC_I32(ir_SHR_I64(ir_MUL_I64(t1, t2), ir_CONST_I64(32)));
        } break;
        case rv_insn_div: {
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_DIV_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
        } break;
        case rv_insn_divu: {
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            ir_ref t1 = ir_BITCAST_U32(t2c_vregs[ir->rs1]);
            ir_ref t2 = ir_BITCAST_U32(t2c_vregs[ir->rs2]);

            t2c_vregs[ir->rd] = ir_DIV_I32(t1, t2);
        } break;
        case rv_insn_rem: {
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_MOD_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
        } break;
        case rv_insn_remu: {
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            ir_ref t1 = ir_BITCAST_U32(t2c_vregs[ir->rs1]);
            ir_ref t2 = ir_BITCAST_U32(t2c_vregs[ir->rs2]);

            t2c_vregs[ir->rd] = ir_MOD_I32(t1, t2);
        } break;
#endif
#if RV32_HAS(EXT_C)
        case rv_insn_cnop:
            break;
        case rv_insn_cli:
        case rv_insn_clui:
            t2c_vregs[ir->rd] = ir_CONST_I32(ir->imm);
            break;
        case rv_insn_cmv:
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] = ir_ADD_I32(t2c_vregs[ir->rs2], ir_CONST_I32(0));
            break;
        case rv_insn_cadd:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_ADD_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_csub:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_SUB_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_cand:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_AND_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_cor:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_OR_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_cxor:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            t2c_vregs[ir->rd] =
                ir_XOR_I32(t2c_vregs[ir->rs1], t2c_vregs[ir->rs2]);
            break;
        case rv_insn_cslli:
            t2c_prepare_operand(ctx, rv, ir->rd);

            t2c_vregs[ir->rd] =
                ir_SHL_I32(t2c_vregs[ir->rd],
                           ir_ZEXT_I32(ir_CONST_I8((uint8_t) ir->imm & 0xff)));
            break;
        case rv_insn_csrli:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rs1] =
                ir_SHR_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->shamt));
            break;
        case rv_insn_csrai:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rs1] =
                ir_SAR_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->shamt));
            break;
        case rv_insn_caddi:
            t2c_prepare_operand(ctx, rv, ir->rd);

            t2c_vregs[ir->rd] = ir_ADD_I32(
                t2c_vregs[ir->rd],
                ir_SEXT_I32(ir_CONST_I16((int16_t) ir->imm & 0xffff)));
            break;
        case rv_insn_candi:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rs1] =
                ir_AND_I32(t2c_vregs[ir->rs1], ir_CONST_I32(ir->imm));
            break;
        case rv_insn_caddi4spn:
            t2c_prepare_operand(ctx, rv, rv_reg_sp);

            t2c_vregs[ir->rd] = ir_ADD_I32(
                t2c_vregs[rv_reg_sp],
                ir_SEXT_I32(ir_CONST_I16((int16_t) ir->imm & 0xffff)));
            break;
        case rv_insn_caddi16sp:
            t2c_prepare_operand(ctx, rv, ir->rd);

            t2c_vregs[ir->rd] =
                ir_ADD_I32(t2c_vregs[ir->rd], ir_CONST_I32(ir->imm));
            break;
        case rv_insn_clwsp:
            t2c_prepare_operand(ctx, rv, rv_reg_sp);

            t2c_vregs[ir->rd] = ir_LOAD_I32(ir_ADD_OFFSET(
                ir_ZEXT_I64(t2c_vregs[rv_reg_sp]), mem_base + ir->imm));
            break;
        case rv_insn_cswsp:
            t2c_prepare_operand(ctx, rv, rv_reg_sp);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(t2c_vregs[rv_reg_sp]),
                                   mem_base + ir->imm),
                     t2c_vregs[ir->rs2]);
            break;
        case rv_insn_clw:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_vregs[ir->rd] = ir_LOAD_I32(ir_ADD_OFFSET(
                ir_ZEXT_I64(t2c_vregs[ir->rs1]), mem_base + ir->imm));
            break;
        case rv_insn_csw:
            t2c_prepare_operand(ctx, rv, ir->rs1);
            t2c_prepare_operand(ctx, rv, ir->rs2);

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(t2c_vregs[ir->rs1]),
                                   mem_base + ir->imm),
                     t2c_vregs[ir->rs2]);
            break;
        case rv_insn_cbeqz:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_store_all_vregs(ctx, rv);

            cond = ir_IF(ir_EQ(t2c_vregs[ir->rs1], ir_CONST_I32(0)));
            t2c_branch_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_cbnez:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_store_all_vregs(ctx, rv);

            cond = ir_IF(ir_NE(t2c_vregs[ir->rs1], ir_CONST_I32(0)));
            t2c_branch_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_cj:
            t2c_store_all_vregs(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_cjal:
            t2c_store_all_vregs(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->X[rv_reg_ra]),
                     ir_CONST_I32(ir->pc + 2));

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc + ir->imm));
            break;
        case rv_insn_cjr:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_store_all_vregs(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), t2c_vregs[ir->rs1]);
            break;
        case rv_insn_cjalr:
            t2c_prepare_operand(ctx, rv, ir->rs1);

            t2c_store_all_vregs(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->X[rv_reg_ra]),
                     ir_CONST_I32(ir->pc + 2));

            ir_STORE(ir_CONST_ADDR(&rv->PC), t2c_vregs[ir->rs1]);
            break;
        case rv_insn_cebreak:
            t2c_store_all_vregs(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(ir->pc));
            ir_CALL_1(IR_UNUSED, ir_CONST_ADDR((uintptr_t) rv->io.on_ebreak),
                      ir_CONST_ADDR(rv));
            break;
#endif
        default:
            printf("Unsupported operator: %d\n", ir->opcode);
            assert(NULL);
        }

        if (!ir->next)
            break;

        ir = ir->next;
    }

    if (t2c_is_indirect_branch(ir->opcode)) {
        ir_RETURN(IR_UNUSED);
        return;
    }

    if (ir->branch_taken) {
        if (set_has(set, ir->branch_taken->pc)) {
            struct ir_map_entry *entry =
                t2c_map_search(loop_map, ir->branch_taken->pc);

            if (entry) {
                t2c_branch_connect(entry, ir_END());
            } else {
                entry = t2c_map_search(merge_map, ir->branch_taken->pc);
                assert(entry);
                t2c_branch_connect(entry, ir_END());
            }
        } else {
            block_t *next_block =
                cache_get(rv->block_cache, ir->branch_taken->pc, false);
            t2c_build(ctx, rv, next_block, set, loop_map, merge_map);
        }

        if (ir->opcode != rv_insn_jal && ir->opcode != rv_insn_cj &&
            ir->opcode != rv_insn_cjal) {
            ir_IF_FALSE(t2c_branch_pop());
        }
    } else {
        if (ir->opcode != rv_insn_jal && ir->opcode != rv_insn_cj &&
            ir->opcode != rv_insn_cjal) {
            if (ir->opcode == rv_insn_cbeqz || ir->opcode == rv_insn_cbnez)
                ir_STORE(ir_CONST_ADDR(&rv->PC),
                         ir_ZEXT_I64(ir_CONST_I32(ir->pc + ir->imm)));
            else
                ir_STORE(
                    ir_CONST_ADDR(&rv->PC),
                    ir_ZEXT_I64(ir_CONST_I32((ir->pc + ir->imm) & 0xfffffffe)));
        }

        ir_RETURN(IR_UNUSED);
        if (ir->opcode != rv_insn_jal && ir->opcode != rv_insn_cj &&
            ir->opcode != rv_insn_cjal) {
            ir_IF_FALSE(t2c_branch_pop());
        }
    }

    if (ir->branch_untaken) {
        if (set_has(set, ir->branch_untaken->pc)) {
            struct ir_map_entry *entry =
                t2c_map_search(loop_map, ir->branch_untaken->pc);

            if (entry) {
                t2c_branch_connect(entry, ir_END());
            } else {
                entry = t2c_map_search(merge_map, ir->branch_untaken->pc);
                assert(entry);
                t2c_branch_connect(entry, ir_END());
            }
        } else {
            block_t *next_block =
                cache_get(rv->block_cache, ir->branch_untaken->pc, false);
            t2c_build(ctx, rv, next_block, set, loop_map, merge_map);
        }
    } else {
        if (ir->opcode != rv_insn_jal && ir->opcode != rv_insn_cj &&
            ir->opcode != rv_insn_cjal) {
            if (ir->opcode == rv_insn_cbeqz || ir->opcode == rv_insn_cbnez)
                ir_STORE(ir_CONST_ADDR(&rv->PC),
                         ir_ZEXT_I64(ir_CONST_I32(ir->pc + 2)));
            else
                ir_STORE(ir_CONST_ADDR(&rv->PC),
                         ir_ZEXT_I64(ir_CONST_I32(ir->pc + 4)));
        }

        if (ir->opcode != rv_insn_jal && ir->opcode != rv_insn_cj &&
            ir->opcode != rv_insn_cjal) {
            ir_RETURN(IR_UNUSED);
        }
    }
}

void t2c_compile(riscv_t *rv, block_t *block)
{
    set_t set;
    struct ir_map loop_map, merge_map;

    proc_cnt++;

    set_reset(&set);
    memset((void *) &loop_map, 0, sizeof(struct ir_map));
    memset((void *) &merge_map, 0, sizeof(struct ir_map));

    ir_ctx *ctx = malloc(sizeof(ir_ctx));

    ir_init(ctx,
            IR_FUNCTION | IR_OPT_INLINE | IR_OPT_FOLDING | IR_OPT_CFG |
                IR_OPT_CODEGEN,
            1024, 4096);
    ctx->ret_type = IR_VOID;

    block->proc_cnt = proc_cnt;
    block->t2_need_merge = false;

    /*
     * The "ir" needs to know the all terminators when building the ir. Thus,
     * we need to find where is the beginning of a loop or the intersection of
     * the control flow.
     */
    t2c_visited_bb_push(block);
    t2c_detect_loop(rv);
    t2c_visited_bb_pop();

    assert(t2c_visited_bb_cnt == 0);

    ir_consistency_check();

    ir_START();

    t2c_build(ctx, rv, block, &set, &loop_map, &merge_map);

    /* merge all terminators to the their destination */
    for (size_t i = 0; i < loop_map.size; i++) {
        assert(loop_map.entries[i].ends_idx > 0);

        ir_ref refs[MAX_T2C_LOOP_TERMINATOR];
        for (size_t j = 0; j < loop_map.entries[i].ends_idx; j++)
            refs[j] = loop_map.entries[i].ends[j];

        ir_MERGE_N(loop_map.entries[i].ends_idx, refs);
        ir_MERGE_SET_OP(loop_map.entries[i].head, 2, ir_LOOP_END());
    }

    for (size_t i = 0; i < merge_map.size; i++) {
        assert(merge_map.entries[i].ends_idx > 0);

        ir_ref refs[MAX_T2C_LOOP_TERMINATOR];
        for (size_t j = 0; j < merge_map.entries[i].ends_idx; j++)
            refs[j] = merge_map.entries[i].ends[j];

        ir_MERGE_N(merge_map.entries[i].ends_idx, refs);
        ir_MERGE_SET_OP(merge_map.entries[i].head, 2, ir_LOOP_END());
    }

    ir_consistency_check();

    size_t size;
    block->func = ir_jit_compile(ctx, 2, &size);
    block->hot2 = true;

    ir_free(ctx);
    free(ctx);
}

/* jit-cache has not been implemented yet */
struct jit_cache *jit_cache_init()
{
    ;
}

void jit_cache_exit(struct jit_cache *cache UNUSED)
{
    ;
}

void jit_cache_clear(struct jit_cache *cache UNUSED)
{
    ;
}
