CC = gcc
CFLAGS = -Wall -Wextra -g -gdwarf-4

TARGET = test
SRC = test.c
BTF = $(TARGET).btf

all: $(TARGET) $(BTF)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

$(BTF): $(TARGET)
	pahole -J $(TARGET) --btf_encode_detached $(BTF)

clean:
	rm -f $(TARGET) $(BTF) *.o

.PHONY: all clean