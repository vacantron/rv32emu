/*
 * rv32emu is freely redistributable under the MIT License. See the file
 * "LICENSE" for information on usage and redistribution of this file.
 */

#pragma once

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

/* UART */

#define IRQ_UART 1
#define IRQ_UART_BIT (1 << IRQ_UART)

typedef struct {
    uint8_t dll, dlh;                  /**< divisor (ignored) */
    uint8_t lcr;                       /**< UART config */
    uint8_t ier;                       /**< interrupt config */
    uint8_t current_int, pending_ints; /**< interrupt status */
    /* other output signals, loopback mode (ignored) */
    uint8_t mcr;
    /* I/O handling */
    int in_fd, out_fd;
    bool in_ready;
} u8250_state_t;
void u8250_update_interrupts(u8250_state_t *uart);
void u8250_check_ready(u8250_state_t *uart);

uint32_t u8250_read(u8250_state_t *uart, uint32_t addr);

void u8250_write(u8250_state_t *uart,
                 uint32_t addr,
                 uint32_t value);

/* create a UART controller */
u8250_state_t *u8250_new();

typedef struct {
    uint32_t masked;
    uint32_t ip;
    uint32_t ie;
    /* state of input interrupt lines (level-triggered), set by environment */
    uint32_t active;
} plic_t;

/* create a PLIC core */
plic_t *plic_new();

typedef struct {
    uint8_t *mem_base;
    uint64_t mem_size;
} memory_t;

/* create a memory instance */
memory_t *memory_new(uint32_t size);

/* delete a memory instance */
void memory_delete(memory_t *m);

/* read an instruction from memory */
uint32_t memory_ifetch(uint32_t addr);

/* read a word from memory */
uint32_t memory_read_w(uint32_t addr);

/* read a short from memory */
uint16_t memory_read_s(uint32_t addr);

/* read a byte from memory */
uint8_t memory_read_b(uint32_t addr);

/* read a length of data from memory */
void memory_read(const memory_t *m, uint8_t *dst, uint32_t addr, uint32_t size);

/* write a length of data to memory */
static inline void memory_write(memory_t *m,
                                uint32_t addr,
                                const uint8_t *src,
                                uint32_t size)
{
    memcpy(m->mem_base + addr, src, size);
}

/* write a word to memory */
void memory_write_w(uint32_t addr, const uint8_t *src);

/* write a short to memory */
void memory_write_s(uint32_t addr, const uint8_t *src);

/* write a byte to memory */
void memory_write_b(uint32_t addr, const uint8_t *src);

/* write a length of certain value to memory */
static inline void memory_fill(memory_t *m,
                               uint32_t addr,
                               uint32_t size,
                               uint8_t val)
{
    memset(m->mem_base + addr, val, size);
}
