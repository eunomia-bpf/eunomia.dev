// Assembly stub to embed fatbin data in .nv_fatbin ELF section
// Using hidden visibility ensures R_X86_64_RELATIVE relocation (not R_X86_64_64),
// which cuobjdump can resolve without a dynamic linker.

    .section .nv_fatbin, "a", @progbits
    .align 8
    .globl _inject_funcs_fatbin_data
    .hidden _inject_funcs_fatbin_data
    .type _inject_funcs_fatbin_data, @object
_inject_funcs_fatbin_data:
    .incbin "inject_funcs.fatbin"
    .size _inject_funcs_fatbin_data, . - _inject_funcs_fatbin_data

    .section .note.GNU-stack, "", @progbits
