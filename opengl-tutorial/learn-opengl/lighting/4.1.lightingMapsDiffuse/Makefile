PWD := $(shell pwd)
CC := gcc
CXX := g++

COMMON := $(PWD)/../../common
CFLAGS :=
CXXFLAGS := -I$(PWD)/../../ -I$(COMMON)
LDFLAGS := -lGL -lGLEW -lglfw -ldl

HEADERCSOURCE := $(wildcard $(COMMON)/*.c)
HEADERCCSOURCE := $(COMMON)/shader_s.cc $(COMMON)/fileSystem.cc $(COMMON)/camera.cc
CCSOURCE := $(wildcard *.cc)
CCEXECUTABLE := $(CCSOURCE:.cc=)
COBJS := $(HEADERCSOURCE:.c=.o)
CCOBJES := $(HEADERCCSOURCE:.cc=.o)

all: $(CCEXECUTABLE)

$(CCEXECUTABLE): $(CCSOURCE)
	$(CXX) -o $@ $@.cc $(COBJS) $(CCOBJES) $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf $(CCEXECUTABLE) *.o