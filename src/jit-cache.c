#include "jit-cache.h"

struct jit_cache *jit_cache_init(size_t size)
{
    struct jit_cache *cache = calloc(size, sizeof(struct jit_cache));
    return cache;
}

void jit_cache_exit(struct jit_cache *cache)
{
    free(cache);
}

void jit_cache_insert(struct jit_cache *cache,
                      size_t size,
                      uint32_t pc,
                      void *entry)
{
    uint32_t pos = pc & (size - 1);

    // override if existing
    cache[pos].pc = pc;
    cache[pos].entry = entry;
}

void jit_cache_clear(struct jit_cache *cache, size_t size)
{
    memset(cache, 0, size * sizeof(struct jit_cache));
}
