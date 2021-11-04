INCDIR=../common
CC=gcc
CXX=g++
LIBS=-lGL -lGLEW -lglfw -lassimp
CFLAGS= -I$(INCDIR)
CCSOURCE = 	$(wildcard $(INCDIR)/*.cc) \
			$(wildcard ./*.cc)

TARGET = tutorial08

OBJ = $(CCSOURCE:.cc=.o)
$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -rf $(TARGET) *.o