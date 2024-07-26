#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define JIT_CACHE_TABLE_SIZE (1 << 12)

struct jit_cache {
    uint32_t pc;
    void *entry;
};

struct jit_cache *jit_cache_init(size_t size);

void jit_cache_exit(struct jit_cache *cache);

void jit_cache_insert(struct jit_cache *cache,
                      size_t size,
                      uint32_t pc,
                      void *entry);

void jit_cache_clear(struct jit_cache *cache, size_t size);
