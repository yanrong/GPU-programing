PWD := $(shell pwd)
CC := gcc
CXX := g++

COMMON := $(PWD)/../../common
CFLAGS :=
CXXFLAGS := -I$(PWD)/../../ -I$(COMMON)
LDFLAGS := -lGL -lGLEW -lglfw -ldl

HEADERCSOURCE := $(wildcard $(COMMON)/*.c)
HEADERCCSOURCE := $(wildcard $(COMMON)/*.cc)
CCSOURCE := $(wildcard *.cc)
CCEXECUTABLE := $(CCSOURCE:.cc=)
COBJS := $(HEADERCSOURCE:.c=.o)
CCOBJES := $(HEADERCCSOURCE:.cc=.o)

all: $(CCEXECUTABLE)

$(COBJS) : $(HEADERCSOURCE)
	$(CC) -c -o $@ $< $(LDFLAGS) $(CFLAGS)

$(CCOBJS) : $(HEADERCCSOURCE)
	$(CXX) -c -o $@ $< $(LDFLAGS) $(CFLAGS)

$(CCEXECUTABLE): $(CCSOURCE) $(COBJS) $(CCOBJES)
	$(CXX) -o $@ $@.cc $(COBJS) $(CCOBJES) $(CXXFLAGS) $(LDFLAGS)

.PHONY: installclean

installclean:
	rm -rf $(CCEXECUTABLE)

.PHONY: clean

clean: installclean
	rm -rf *.o $(COMMON)/*.o