PWD := $(shell pwd)
CC := gcc
CXX := g++
LIBS := -lGL -lGLEW -lglfw -ldl

CFLAGS :=
CXXFLAGS :=
LDFLAGS :=

COMMON := $(PWD)/../common
CSOURCE := $(wildcard $(COMMON)/*.c)
CCSOURCE := $(wildcard *.cc)
CCEXECUTABLE := $(CCSOURCE:.cc=)
COBJS = $(CSOURCE:.c=.o)

all: $(CCEXECUTABLE)

$(COBJS) : $(CSOURCE)
	$(CC) -c -o $@ $< $(LDFLAGS) $(CFLAGS)

$(CCEXECUTABLE): $(CCSOURCE) $(COBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -rf $(CCEXECUTABLE) *.o $(COMMON)/*.o