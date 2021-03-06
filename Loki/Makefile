# Makefile that builds all the isa tests then archives them for later testing.

# User configurable parameters.
# Temporary directory.
BUILD_DIR ?= build
# Output directory.
OUT_DIR ?= bin
# Root libloki directory.
LIBLOKI_DIR ?=

OPTIMIZATION = -O3
# Tweakables.
# Variable containing list of all input test programs.
TESTS := $(wildcard src/*.c)
# Ignore disabled tests
TESTS := $(filter-out $(shell cat disabled),$(TESTS))
# Variable containing list of all compiled test programs.
ELFS := $(patsubst src/%.c,$(BUILD_DIR)/%.elf,$(TESTS))

# Make archive by default.
.PHONY: all
all: $(OUT_DIR)/test-libloki.tar.gz

# Rule to archive the elf files.
$(OUT_DIR)/test-libloki.tar.gz: $(ELFS) | $(OUT_DIR)
	tar -czf $@ -C $(BUILD_DIR) $(patsubst $(BUILD_DIR)/%,%,$(ELFS))

# Rule to build one elf file.
$(BUILD_DIR)/%.elf: src/%.c | $(BUILD_DIR)
	loki-clang -ccc-host-triple loki-elf-linux -I$(LIBLOKI_DIR)/include -L$(LIBLOKI_DIR)/lib $(OPTIMIZATION) -o $@ $< -lloki -lm -DLOKI

# Rule to make intermediate directories.
$(OUT_DIR):
	mkdir $@
$(BUILD_DIR):
	mkdir $@

# Rule to clean regeneratable directories.
.PHONY: clean
clean:
	rm -rf $(wildcard $(OUT_DIR))
	rm -rf $(wildcard $(BUILD_DIR))

