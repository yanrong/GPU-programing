CC := gcc
CXX := g++

LIBS := -lGL -lGLEW -lglut

PWD := $(shell pwd)
INCDIR := $(PWD)/../include
COMMON := $(PWD)/../common
CXXFLAGS := -I$(PWD)/.. -I$(INCDIR)

SRCFILES := $(wildcard *.cpp)

OBJS := $(SRCFILES:.cpp=.o)
TARGET := tutorial01

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LIBS)

.PHONY: clean

clean:
	rm -rf $(TARGET) *.o $(COMMON)/*.o