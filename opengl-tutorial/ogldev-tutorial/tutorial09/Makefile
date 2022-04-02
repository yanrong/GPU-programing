CC := gcc
CXX := g++
LIBS := -lGL -lGLEW -lglfw -lassimp

PWD := $(shell pwd)
INCDIR := $(PWD)/../common

CXXFLAGS := -I$(PWD)/.. -I$(INCDIR)
CCSOURCE := $(wildcard $(INCDIR)/*.cc) \
			$(wildcard ./*.cc)

TARGET := tutorial08

OBJS := $(CCSOURCE:.cc=.o)
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -rf $(TARGET) *.o $(INCDIR)/*.o