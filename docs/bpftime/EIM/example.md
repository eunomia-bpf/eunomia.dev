# EIM: Example code

See the source for this page to get example code for the EIM.

## build the example

```bash
make
```

## See the types defined for the example

```bash
$ sudo bpftool btf dump file /home/yunwei37/bpftime-evaluation/interfeace/test.btf
[1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
[2] TYPEDEF 'size_t' type_id=3
[3] INT 'long unsigned int' size=8 bits_offset=0 nr_bits=64 encoding=(none)
[4] INT 'unsigned int' size=4 bits_offset=0 nr_bits=32 encoding=(none)
[5] PTR '(anon)' type_id=0
[6] INT 'unsigned char' size=1 bits_offset=0 nr_bits=8 encoding=(none)
[7] INT 'short unsigned int' size=2 bits_offset=0 nr_bits=16 encoding=(none)
[8] INT 'signed char' size=1 bits_offset=0 nr_bits=8 encoding=SIGNED
[9] INT 'short int' size=2 bits_offset=0 nr_bits=16 encoding=SIGNED
[10] INT 'long int' size=8 bits_offset=0 nr_bits=64 encoding=SIGNED
[11] TYPEDEF '__ssize_t' type_id=10
[12] PTR '(anon)' type_id=13
[13] INT 'char' size=1 bits_offset=0 nr_bits=8 encoding=SIGNED
[14] CONST '(anon)' type_id=13
[15] PTR '(anon)' type_id=14
[16] TYPEDEF 'ssize_t' type_id=11
[17] TYPEDEF '__bpftime_ext_ufunction_file_ops_host_open_file' type_id=1
[18] TYPEDEF '__bpftime_ext_ufunction_file_ops_host_read_file' type_id=1
[19] TYPEDEF '__bpftime_ext_ufunction_file_ops_host_write_file' type_id=1
[20] TYPEDEF '__bpftime_capability_FileRead_includes_file_ops_host_open_file' type_id=1
[21] TYPEDEF '__bpftime_capability_FileRead_includes_file_ops_host_read_file' type_id=1
[22] TYPEDEF '__bpftime_capability_FileWrite_includes_file_ops_host_write_file' type_id=1
[23] TYPEDEF '__bpftime_role_BasicFileRole_has_cap_FileRead' type_id=1
[24] TYPEDEF '__bpftime_role_ExtendedFileRole_has_cap_FileWrite' type_id=1
[25] TYPEDEF '__bpftime_role_ExtendedFileRole_inherits_BasicFileRole' type_id=1
[26] TYPEDEF '__bpftime_role_AdminRole' type_id=1
[27] TYPEDEF '__bpftime_role_AdminRole_inherits_ExtendedFileRole' type_id=1
[28] PTR '(anon)' type_id=29
[29] CONST '(anon)' type_id=0
[30] FUNC_PROTO '(anon)' ret_type_id=1 vlen=0
[31] FUNC 'main' type_id=30 linkage=static
[32] FUNC_PROTO '(anon)' ret_type_id=1 vlen=2
        'data' type_id=15
        'length' type_id=1
[33] FUNC '__bpftime_extension_entry_AdminRole_analyze_data' type_id=32 linkage=static
[34] FUNC_PROTO '(anon)' ret_type_id=1 vlen=2
        'input' type_id=15
        'mode' type_id=1
[35] FUNC '__bpftime_extension_entry_AdminRole_process_string' type_id=34 linkage=static
[36] FUNC_PROTO '(anon)' ret_type_id=1 vlen=3
        'fd' type_id=1
        'buffer' type_id=15
        'length' type_id=1
[37] FUNC 'host_write_file' type_id=36 linkage=static
[38] FUNC_PROTO '(anon)' ret_type_id=1 vlen=3
        'fd' type_id=1
        'buffer' type_id=12
        'length' type_id=1
[39] FUNC 'host_read_file' type_id=38 linkage=static
[40] FUNC_PROTO '(anon)' ret_type_id=1 vlen=2
        'filename' type_id=15
        'flags' type_id=1
[41] FUNC 'host_open_file' type_id=40 linkage=static
```

In BTF, we use typename to match the spec between host and extension.

We can also mannually define the types with yaml for json config.