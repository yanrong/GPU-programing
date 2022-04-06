CC := gcc
CXX := g++

LIBS := -lGL -lGLEW -lglut

PWD := $(shell pwd)
INCDIR := $(PWD)/../include
COMMONDIR := $(PWD)/../common

CXXFLAGS := -I$(PWD)/.. -I$(INCDIR)
#CCSOURCE := $(wildcard $(INCDIR)/*.cpp) \
#			$(wildcard ./*.cpp)

SRCFILES := $(wildcard  *.cpp)
OBJS := $(SRCFILES:.cpp=.o)

TARGET := tutorial02

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -rf $(TARGET) *.o $(COMMONDIR)/*.o