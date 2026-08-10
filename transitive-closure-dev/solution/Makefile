NVCC ?= nvcc
BUILD_DIR ?= ../build/submission
API_DIR ?= ../interface

TARGET := $(BUILD_DIR)/libclosure_submission.so
SOURCES := $(wildcard src/*.cu)
HEADERS := $(wildcard include/*) $(API_DIR)/closure_api.h
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_89 -lineinfo

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler -fPIC \
		-I$(API_DIR) -Iinclude $(SOURCES) -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR)
