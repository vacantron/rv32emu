#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "codegen-amd64.c"

int main()
{
    codegen_buf_t buf = {.capacity = 0xffff, .size = 0};

    buf.insns = calloc(1, 4 * 0xffff);

    emit_const(&buf, R13, 0x1234567822345678);

    return 0;
}
