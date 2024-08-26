/*
 * rv32emu is freely redistributable under the MIT License. See the file
 * "LICENSE" for information on usage and redistribution of this file.
 */

#include <assert.h>

#include <ir.h>
#include <ir_builder.h>

#include "riscv_private.h"

#define MAX_IR_MAP_SIZE (1 << 8)
#define MAX_IR_STACK_SIZE (1 << 8)
#define MAX_IR_LOOP_END (1 << 8)
#define MAX_VISITED_BLOCK (1 << 8)

static ir_ref ir_reg_refs[N_RV_REGS];

static ir_ref ir_branch_stack[MAX_IR_STACK_SIZE];
static int ir_branch_stack_idx = 0;

static int proc_cnt = 0;

static struct ir_map_entry {
    uint32_t pc;
    ir_ref head;
    ir_ref ends[MAX_IR_LOOP_END];
    size_t end_idx;
};

static void ir_branch_stack_push(ir_ref e)
{
    ir_branch_stack[ir_branch_stack_idx++] = e;
    assert(ir_branch_stack_idx <= MAX_IR_STACK_SIZE);
}

static ir_ref ir_branch_stack_pop()
{
    assert(ir_branch_stack_idx > 0);
    return ir_branch_stack[--ir_branch_stack_idx];
}

static struct ir_map {
    size_t size;
    struct ir_map_entry entries[MAX_IR_MAP_SIZE];
};

static void ir_map_insert(struct ir_map *map, uint32_t pc, ir_ref head)
{
    struct ir_map_entry entry = {.pc = pc, .head = head, .end_idx = 0};
    map->entries[map->size++] = entry;

    assert(map->size <= MAX_IR_MAP_SIZE);
}

static struct ir_map_entry *ir_map_search(struct ir_map *map, uint32_t pc)
{
    for (size_t i = 0; i < map->size; i++) {
        if (map->entries[i].pc == pc)
            return &map->entries[i];
    }
    return NULL;
}

static void ir_map_element_insert(struct ir_map_entry *entry, ir_ref e)
{
    entry->ends[entry->end_idx++] = e;
    assert(entry->end_idx <= MAX_IR_LOOP_END);
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

        ir_STORE(ir_CONST_ADDR(&rv->X[i]),
                 ir_TRUNC_I32(ir_ZEXT_I64(ir_reg_refs[i])));
    }
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


static block_t *ir_visited_bb_stack[MAX_VISITED_BLOCK];
static int ir_visited_bb_stack_idx;

static void ir_visited_bb_stack_push(block_t *block)
{
    ir_visited_bb_stack[ir_visited_bb_stack_idx++] = block;
    assert(ir_visited_bb_stack_idx <= MAX_VISITED_BLOCK);
}

static block_t *ir_visited_bb_stack_pop()
{
    assert(ir_visited_bb_stack_idx > 0);
    return ir_visited_bb_stack[--ir_visited_bb_stack_idx];
}

static block_t *ir_visited_bb_stack_peek()
{
    return ir_visited_bb_stack[ir_visited_bb_stack_idx - 1];
}

static bool ir_bb_in_visited_stack(block_t *block)
{
    for (int i = 0; i < ir_visited_bb_stack_idx; i++) {
        if (ir_visited_bb_stack[i] == block)
            return true;
    }
    return false;
}

static void ir_detect_loop(riscv_t *rv)
{
    block_t *block = ir_visited_bb_stack_peek();
    rv_insn_t *tail = block->ir_tail;

    if (ir_is_terminal_insn(tail->opcode))
        return;

    if (tail->branch_taken) {
        block_t *next =
            cache_get(rv->block_cache, tail->branch_taken->pc, false);
        if (next->proc_cnt != proc_cnt) {
            next->proc_cnt = proc_cnt;
            next->ir_has_loops = false;
            next->ir_need_merge = false;
            next->visited_by = block;
        } else {
            if (next->visited_by != block) {
                next->ir_need_merge = true;
            }
        }

        if (!ir_bb_in_visited_stack(next)) {
            ir_visited_bb_stack_push(next);
            ir_detect_loop(rv);
            ir_visited_bb_stack_pop();
        } else {
            next->ir_has_loops = true;
        }
    }

    if (tail->branch_untaken) {
        block_t *next =
            cache_get(rv->block_cache, tail->branch_untaken->pc, false);
        if (next->proc_cnt != proc_cnt) {
            next->proc_cnt = proc_cnt;
            next->ir_has_loops = false;
            next->ir_need_merge = false;
            next->visited_by = block;
        } else {
            if (next->visited_by != block) {
                next->ir_need_merge = true;
            }
        }

        if (!ir_bb_in_visited_stack(next)) {
            ir_visited_bb_stack_push(next);
            ir_detect_loop(rv);
            ir_visited_bb_stack_pop();
        } else {
            next->ir_has_loops = true;
        }
    }
}

static inline void ir_prepare_operand(ir_ctx *ctx, riscv_t *rv, uint8_t idx)
{
    if (ir_reg_refs[idx]) {
        ir_reg_refs[idx] = ir_TRUNC_I32(ir_ZEXT_I64(ir_reg_refs[idx]));
        return;
    }

    ir_reg_refs[idx] = ir_LOAD_I32(ir_CONST_ADDR(&rv->X[idx]));
}

void ir_build(ir_ctx *ctx,
              riscv_t *rv,
              block_t *block,
              set_t *set,
              struct ir_map *loop_map,
              struct ir_map *merge_map)
{
    rv_insn_t *insn = block->ir_head;
    uintptr_t mem_base = (uintptr_t) PRIV(rv)->mem->mem_base;

    if (set_has(set, insn->pc))
        return;

    set_add(set, insn->pc);

    if (block->proc_cnt == proc_cnt) {
        if (block->ir_has_loops) {
            ir_map_insert(loop_map, insn->pc, ir_LOOP_BEGIN(ir_END()));
        } else if (block->ir_need_merge) {
            /* modify source code of src/ir to use ir_LOOP_BEGIN instead of
             * ir_BLOCK_BEGIN
             */
            ir_map_insert(merge_map, insn->pc, ir_LOOP_BEGIN(ir_END()));
        }
    }

    ir_reg_refs_clear();

    while (1) {
        ir_ref cond;

        switch (insn->opcode) {
        case rv_insn_nop:
            break;
        case rv_insn_auipc:
            /* incorrect immediate value fixed */
            ir_reg_refs[insn->rd] = ir_CONST_I32(insn->imm + insn->pc);
            break;
        case rv_insn_lui:
            ir_reg_refs[insn->rd] = ir_CONST_I32(insn->imm);
            break;
        case rv_insn_add:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_ADD_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_sub:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_SUB_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_xor:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_XOR_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_or:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_OR_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_and:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_AND_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_sll:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] = ir_SHL_I32(
                ir_reg_refs[insn->rs1],
                ir_AND_I32(ir_reg_refs[insn->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_srl:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] = ir_SHR_I32(
                ir_reg_refs[insn->rs1],
                ir_AND_I32(ir_reg_refs[insn->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_sra:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] = ir_SAR_I32(
                ir_reg_refs[insn->rs1],
                ir_AND_I32(ir_reg_refs[insn->rs2], ir_CONST_I32(0x1f)));
            break;
        case rv_insn_addi:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] =
                ir_ADD_I32(ir_reg_refs[insn->rs1], ir_CONST_I32(insn->imm));
            break;
        case rv_insn_andi:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] =
                ir_AND_I32(ir_reg_refs[insn->rs1], ir_CONST_I32(insn->imm));
            break;
        case rv_insn_ori:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] =
                ir_OR_I32(ir_reg_refs[insn->rs1], ir_CONST_I32(insn->imm));
            break;
        case rv_insn_xori:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] =
                ir_XOR_I32(ir_reg_refs[insn->rs1], ir_CONST_I32(insn->imm));
            break;
        case rv_insn_slli:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_SHL_I32(ir_reg_refs[insn->rs1],
                                               ir_CONST_I32(insn->imm & 0x1f));
            break;
        case rv_insn_srli:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_SHR_I32(ir_reg_refs[insn->rs1],
                                               ir_CONST_I32(insn->imm & 0x1f));
            break;
        case rv_insn_srai:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_SAR_I32(ir_reg_refs[insn->rs1],
                                               ir_CONST_I32(insn->imm & 0x1f));
            break;
        case rv_insn_lw:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_LOAD_I32(ir_ADD_OFFSET(
                ir_ZEXT_I64(ir_reg_refs[insn->rs1]), mem_base + insn->imm));
            break;
        case rv_insn_lh:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_SEXT_I32(ir_LOAD_I16(ir_ADD_OFFSET(
                ir_ZEXT_I64(ir_reg_refs[insn->rs1]), mem_base + insn->imm)));
            break;
        case rv_insn_lb:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_SEXT_I32(ir_LOAD_I8(ir_ADD_OFFSET(
                ir_ZEXT_I64(ir_reg_refs[insn->rs1]), mem_base + insn->imm)));
            break;
        case rv_insn_lhu:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_ZEXT_I32(ir_LOAD_I16(ir_ADD_OFFSET(
                ir_ZEXT_I64(ir_reg_refs[insn->rs1]), mem_base + insn->imm)));
            break;
        case rv_insn_lbu:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_ZEXT_I32(ir_LOAD_I8(ir_ADD_OFFSET(
                ir_ZEXT_I64(ir_reg_refs[insn->rs1]), mem_base + insn->imm)));
            break;
        case rv_insn_sw:
            ir_prepare_operand(ctx, rv, insn->rs1);

            if (!ir_reg_refs[insn->rs2]) {
                ir_reg_refs[insn->rs2] =
                    ir_LOAD_I32(ir_CONST_ADDR(&rv->X[insn->rs2]));
            } else {
                ir_reg_refs[insn->rs2] =
                    ir_TRUNC_I32(ir_ZEXT_I64(ir_reg_refs[insn->rs2]));
            }

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(ir_reg_refs[insn->rs1]),
                                   mem_base + insn->imm),
                     ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_sh:
            ir_prepare_operand(ctx, rv, insn->rs1);

            if (!ir_reg_refs[insn->rs2]) {
                ir_reg_refs[insn->rs2] =
                    ir_LOAD_I16(ir_CONST_ADDR(&rv->X[insn->rs2]));
            } else {
                ir_STORE(ir_CONST_ADDR(&rv->X[insn->rs2]),
                         ir_reg_refs[insn->rs2]);
                ir_reg_refs[insn->rs2] = ir_TRUNC_I16(ir_reg_refs[insn->rs2]);
            }

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(ir_reg_refs[insn->rs1]),
                                   mem_base + insn->imm),
                     ir_reg_refs[insn->rs2]);

            /* abandon 16-bit value */
            ir_reg_refs[insn->rs2] = 0;
            break;
        case rv_insn_sb:
            ir_prepare_operand(ctx, rv, insn->rs1);

            if (!ir_reg_refs[insn->rs2]) {
                ir_reg_refs[insn->rs2] =
                    ir_LOAD_I8(ir_CONST_ADDR(&rv->X[insn->rs2]));
            } else {
                ir_STORE(ir_CONST_ADDR(&rv->X[insn->rs2]),
                         ir_reg_refs[insn->rs2]);
                ir_reg_refs[insn->rs2] = ir_TRUNC_I8(ir_reg_refs[insn->rs2]);
            }

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(ir_reg_refs[insn->rs1]),
                                   mem_base + insn->imm),
                     ir_reg_refs[insn->rs2]);

            /* abandon 8-bit value */
            ir_reg_refs[insn->rs2] = 0;
            break;
        case rv_insn_slt:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_COND_I32(ir_ZEXT_I32(ir_LT(ir_reg_refs[insn->rs1],
                                              ir_reg_refs[insn->rs2])),
                            ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_sltu:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_COND_I32(ir_ZEXT_I32(ir_ULT(ir_reg_refs[insn->rs1],
                                               ir_reg_refs[insn->rs2])),
                            ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_slti:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] =
                ir_COND_I32(ir_ZEXT_I32(ir_LT(ir_reg_refs[insn->rs1],
                                              ir_CONST_I32(insn->imm))),
                            ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_sltiu:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] =
                ir_COND_I32(ir_ZEXT_I32(ir_ULT(ir_reg_refs[insn->rs1],
                                               ir_CONST_I32(insn->imm))),
                            ir_CONST_I32(1), ir_CONST_I32(0));
            break;
        case rv_insn_beq:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_EQ(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]));
            ir_branch_stack_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_bne:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_NE(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]));
            ir_branch_stack_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_bge:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_GE(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]));
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            ir_branch_stack_push(cond);
            break;
        case rv_insn_blt:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_LT(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]));
            ir_branch_stack_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_bgeu:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond =
                ir_IF(ir_UGE(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]));
            ir_branch_stack_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_bltu:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs_store_all(ctx, rv);

            cond =
                ir_IF(ir_ULT(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]));
            ir_branch_stack_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_jal:
            ir_reg_refs_store_all(ctx, rv);

            if (insn->rd) {
                ir_STORE(ir_CONST_ADDR(&rv->X[insn->rd]),
                         ir_CONST_I32(insn->pc + 4));
            }

            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_jalr:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs_store_all(ctx, rv);

            if (insn->rd) {
                ir_STORE(ir_CONST_ADDR(&rv->X[insn->rd]),
                         ir_CONST_I32(insn->pc + 4));
            }

            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_ADD_I32(ir_reg_refs[insn->rs1],
                                ir_CONST_I32(insn->imm & ~1U)));
            break;
        case rv_insn_ecall:
            ir_reg_refs_store_all(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(insn->pc));
            ir_CALL_1(IR_UNUSED, ir_CONST_ADDR((uintptr_t) rv->io.on_ecall),
                      ir_CONST_ADDR(rv));
            break;
        case rv_insn_ebreak:
            ir_reg_refs_store_all(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(insn->pc));
            ir_CALL_1(IR_UNUSED, ir_CONST_ADDR((uintptr_t) rv->io.on_ebreak),
                      ir_CONST_ADDR(rv));
            break;
#if RV32_HAS(EXT_M)
        case rv_insn_mul:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_MUL_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_mulh: {
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_ref t1 = ir_SEXT_I64(ir_reg_refs[insn->rs1]);
            ir_ref t2 = ir_SEXT_I64(ir_reg_refs[insn->rs2]);

            ir_reg_refs[insn->rd] =
                ir_TRUNC_I32(ir_SHR_I64(ir_MUL_I64(t1, t2), ir_CONST_I64(32)));
        } break;
        case rv_insn_mulhsu: {
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_ref t1 = ir_SEXT_I64(ir_reg_refs[insn->rs1]);
            ir_ref t2 = ir_ZEXT_I64(ir_reg_refs[insn->rs2]);

            ir_reg_refs[insn->rd] =
                ir_TRUNC_I32(ir_SHR_I64(ir_MUL_I64(t1, t2), ir_CONST_I64(32)));
        } break;
        case rv_insn_mulhu: {
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_ref t1 = ir_ZEXT_I64(ir_reg_refs[insn->rs1]);
            ir_ref t2 = ir_ZEXT_I64(ir_reg_refs[insn->rs2]);

            ir_reg_refs[insn->rd] =
                ir_TRUNC_I32(ir_SHR_I64(ir_MUL_I64(t1, t2), ir_CONST_I64(32)));
        } break;
        case rv_insn_div: {
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_DIV_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
        } break;
        case rv_insn_divu: {
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_ref t1 = ir_BITCAST_U32(ir_reg_refs[insn->rs1]);
            ir_ref t2 = ir_BITCAST_U32(ir_reg_refs[insn->rs2]);

            ir_reg_refs[insn->rd] = ir_DIV_I32(t1, t2);
        } break;
        case rv_insn_rem: {
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_MOD_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
        } break;
        case rv_insn_remu: {
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_ref t1 = ir_BITCAST_U32(ir_reg_refs[insn->rs1]);
            ir_ref t2 = ir_BITCAST_U32(ir_reg_refs[insn->rs2]);

            ir_reg_refs[insn->rd] = ir_MOD_I32(t1, t2);
        } break;
#endif
#if RV32_HAS(EXT_C)
        case rv_insn_cnop:
            break;
        case rv_insn_cli:
        case rv_insn_clui:
            ir_reg_refs[insn->rd] = ir_CONST_I32(insn->imm);
            break;
        case rv_insn_cmv:
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_ADD_I32(ir_reg_refs[insn->rs2], ir_CONST_I32(0));
            break;
        case rv_insn_cadd:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_ADD_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_csub:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_SUB_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_cand:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_AND_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_cor:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_OR_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_cxor:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_reg_refs[insn->rd] =
                ir_XOR_I32(ir_reg_refs[insn->rs1], ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_cslli:
            ir_prepare_operand(ctx, rv, insn->rd);

            ir_reg_refs[insn->rd] = ir_SHL_I32(
                ir_reg_refs[insn->rd],
                ir_ZEXT_I32(ir_CONST_I8((uint8_t) insn->imm & 0xff)));
            break;
        case rv_insn_csrli:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rs1] =
                ir_SHR_I32(ir_reg_refs[insn->rs1], ir_CONST_I32(insn->shamt));
            break;
        case rv_insn_csrai:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rs1] =
                ir_SAR_I32(ir_reg_refs[insn->rs1], ir_CONST_I32(insn->shamt));
            break;
        case rv_insn_caddi:
            ir_prepare_operand(ctx, rv, insn->rd);

            ir_reg_refs[insn->rd] = ir_ADD_I32(
                ir_reg_refs[insn->rd],
                ir_SEXT_I32(ir_CONST_I16((int16_t) insn->imm & 0xffff)));
            break;
        case rv_insn_candi:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rs1] =
                ir_AND_I32(ir_reg_refs[insn->rs1], ir_CONST_I32(insn->imm));
            break;
        case rv_insn_caddi4spn:
            ir_prepare_operand(ctx, rv, rv_reg_sp);

            ir_reg_refs[insn->rd] = ir_ADD_I32(
                ir_reg_refs[rv_reg_sp],
                ir_SEXT_I32(ir_CONST_I16((int16_t) insn->imm & 0xffff)));
            break;
        case rv_insn_caddi16sp:
            ir_prepare_operand(ctx, rv, insn->rd);

            ir_reg_refs[insn->rd] =
                ir_ADD_I32(ir_reg_refs[insn->rd], ir_CONST_I32(insn->imm));
            break;
        case rv_insn_clwsp:
            ir_prepare_operand(ctx, rv, rv_reg_sp);

            ir_reg_refs[insn->rd] = ir_LOAD_I32(ir_ADD_OFFSET(
                ir_ZEXT_I64(ir_reg_refs[rv_reg_sp]), mem_base + insn->imm));
            break;
        case rv_insn_cswsp:
            ir_prepare_operand(ctx, rv, rv_reg_sp);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(ir_reg_refs[rv_reg_sp]),
                                   mem_base + insn->imm),
                     ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_clw:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs[insn->rd] = ir_LOAD_I32(ir_ADD_OFFSET(
                ir_ZEXT_I64(ir_reg_refs[insn->rs1]), mem_base + insn->imm));
            break;
        case rv_insn_csw:
            ir_prepare_operand(ctx, rv, insn->rs1);
            ir_prepare_operand(ctx, rv, insn->rs2);

            ir_STORE(ir_ADD_OFFSET(ir_ZEXT_I64(ir_reg_refs[insn->rs1]),
                                   mem_base + insn->imm),
                     ir_reg_refs[insn->rs2]);
            break;
        case rv_insn_cbeqz:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_EQ(ir_reg_refs[insn->rs1], ir_CONST_I32(0)));
            ir_branch_stack_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_cbnez:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs_store_all(ctx, rv);

            cond = ir_IF(ir_NE(ir_reg_refs[insn->rs1], ir_CONST_I32(0)));
            ir_branch_stack_push(cond);
            ir_IF_TRUE(cond);
            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_cj:
            ir_reg_refs_store_all(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_cjal:
            ir_reg_refs_store_all(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->X[rv_reg_ra]),
                     ir_CONST_I32(insn->pc + 2));

            ir_STORE(ir_CONST_ADDR(&rv->PC),
                     ir_CONST_I32(insn->pc + insn->imm));
            break;
        case rv_insn_cjr:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs_store_all(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_reg_refs[insn->rs1]);
            break;
        case rv_insn_cjalr:
            ir_prepare_operand(ctx, rv, insn->rs1);

            ir_reg_refs_store_all(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->X[rv_reg_ra]),
                     ir_CONST_I32(insn->pc + 2));

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_reg_refs[insn->rs1]);
            break;
        case rv_insn_cebreak:
            ir_reg_refs_store_all(ctx, rv);

            ir_STORE(ir_CONST_ADDR(&rv->PC), ir_CONST_I32(insn->pc));
            ir_CALL_1(IR_UNUSED, ir_CONST_ADDR((uintptr_t) rv->io.on_ebreak),
                      ir_CONST_ADDR(rv));
            break;
#endif
        default:
            printf("Unsupported operator: %d\n", insn->opcode);
            assert(NULL);
        }

        if (!insn->next)
            break;

        insn = insn->next;
    }

    if (ir_is_terminal_insn(insn->opcode)) {
        ir_RETURN(IR_UNUSED);
        return;
    }

    if (insn->branch_taken) {
        if (set_has(set, insn->branch_taken->pc)) {
            struct ir_map_entry *entry =
                ir_map_search(loop_map, insn->branch_taken->pc);

            if (entry) {
                ir_map_element_insert(entry, ir_END());
            } else {
                entry = ir_map_search(merge_map, insn->branch_taken->pc);
                assert(entry);
                ir_map_element_insert(entry, ir_END());
            }
        } else {
            block_t *next_block =
                cache_get(rv->block_cache, insn->branch_taken->pc, false);
            ir_build(ctx, rv, next_block, set, loop_map, merge_map);
        }

        if (insn->opcode != rv_insn_jal && insn->opcode != rv_insn_cj &&
            insn->opcode != rv_insn_cjal) {
            ir_IF_FALSE(ir_branch_stack_pop());
        }
    } else {
        if (insn->opcode != rv_insn_jal && insn->opcode != rv_insn_cj &&
            insn->opcode != rv_insn_cjal) {
            if (insn->opcode == rv_insn_cbeqz || insn->opcode == rv_insn_cbnez)
                ir_STORE(ir_CONST_ADDR(&rv->PC),
                         ir_ZEXT_I64(ir_CONST_I32(insn->pc + insn->imm)));
            else
                ir_STORE(ir_CONST_ADDR(&rv->PC),
                         ir_ZEXT_I64(ir_CONST_I32((insn->pc + insn->imm) &
                                                  0xfffffffe)));
        }

        ir_RETURN(IR_UNUSED);
        if (insn->opcode != rv_insn_jal && insn->opcode != rv_insn_cj &&
            insn->opcode != rv_insn_cjal) {
            ir_IF_FALSE(ir_branch_stack_pop());
        }
    }

    if (insn->branch_untaken) {
        if (set_has(set, insn->branch_untaken->pc)) {
            struct ir_map_entry *entry =
                ir_map_search(loop_map, insn->branch_untaken->pc);

            if (entry) {
                ir_map_element_insert(entry, ir_END());
            } else {
                entry = ir_map_search(merge_map, insn->branch_untaken->pc);
                assert(entry);
                ir_map_element_insert(entry, ir_END());
            }
        } else {
            block_t *next_block =
                cache_get(rv->block_cache, insn->branch_untaken->pc, false);
            ir_build(ctx, rv, next_block, set, loop_map, merge_map);
        }
    } else {
        if (insn->opcode != rv_insn_jal && insn->opcode != rv_insn_cj &&
            insn->opcode != rv_insn_cjal) {
            if (insn->opcode == rv_insn_cbeqz || insn->opcode == rv_insn_cbnez)
                ir_STORE(ir_CONST_ADDR(&rv->PC),
                         ir_ZEXT_I64(ir_CONST_I32(insn->pc + 2)));
            else
                ir_STORE(ir_CONST_ADDR(&rv->PC),
                         ir_ZEXT_I64(ir_CONST_I32(insn->pc + 4)));
        }

        if (insn->opcode != rv_insn_jal && insn->opcode != rv_insn_cj &&
            insn->opcode != rv_insn_cjal) {
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
    block->ir_need_merge = false;

    ir_visited_bb_stack_push(block);
    ir_detect_loop(rv);
    ir_visited_bb_stack_pop();

    assert(ir_visited_bb_stack_idx == 0);

    ir_consistency_check();

    ir_START();

    ir_build(ctx, rv, block, &set, &loop_map, &merge_map);

    for (size_t i = 0; i < loop_map.size; i++) {
        assert(loop_map.entries[i].end_idx > 0);

        ir_ref refs[MAX_IR_LOOP_END];
        for (size_t j = 0; j < loop_map.entries[i].end_idx; j++)
            refs[j] = loop_map.entries[i].ends[j];

        ir_MERGE_N(loop_map.entries[i].end_idx, refs);
        ir_MERGE_SET_OP(loop_map.entries[i].head, 2, ir_LOOP_END());
    }

    for (size_t i = 0; i < merge_map.size; i++) {
        assert(merge_map.entries[i].end_idx > 0);

        ir_ref refs[MAX_IR_LOOP_END];
        for (size_t j = 0; j < merge_map.entries[i].end_idx; j++)
            refs[j] = merge_map.entries[i].ends[j];

        ir_MERGE_N(merge_map.entries[i].end_idx, refs);
        /* need to modify source code of src/ir  */
        ir_MERGE_SET_OP(merge_map.entries[i].head, 2, ir_LOOP_END());
    }

    ir_consistency_check();

    size_t size;
    block->func = ir_jit_compile(ctx, 2, &size);
    block->hot2 = true;

    ir_free(ctx);
    free(ctx);
}

struct jit_cache *jit_cache_init()
{
    ;
}

void jit_cache_exit(struct jit_cache *cache)
{
    ;
}

void jit_cache_clear(struct jit_cache *cache)
{
    ;
}
