PWD := $(shell pwd)
CC := gcc
CXX := g++

CFLAGS :=
CXXFLAGS :=
LDFLAGS := -lGL -lGLEW -lglfw -ldl

COMMON := $(PWD)/../common
CSOURCE := $(wildcard $(COMMON)/*.c)
CCSOURCE := $(wildcard *.cc)
CCEXECUTABLE := $(CCSOURCE:.cc=)
COBJS := $(CSOURCE:.c=.o)

all: $(COBJS) $(CCEXECUTABLE)

$(COBJS) : $(CSOURCE)
	$(CC) -c -o $@ $< $(LDFLAGS) $(CFLAGS)

$(CCEXECUTABLE): $(CCSOURCE)
	$(CXX) -o $@ $@.cc $(COBJS) $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf $(CCEXECUTABLE) *.o $(COMMON)/*.o