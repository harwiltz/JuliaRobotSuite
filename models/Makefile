SRCS=$(wildcard *.urdf.xacro)
TARGETS=$(patsubst %.urdf.xacro,%.urdf,$(SRCS))

XACRO=xacro

all: $(TARGETS)

%.urdf: %.urdf.xacro
	$(XACRO) -o $@ $^

clean:
	rm -rf $(TARGETS)

.PHONY: all clean
