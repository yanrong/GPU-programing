CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

VulkanTest: vulkanTest.cpp
	g++ $(CFLAGS) -o vulkanTest vulkanTest.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTest
	./vulkanTest

clean:
	rm -f vulkanTest
