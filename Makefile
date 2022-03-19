all: models

models:
	$(MAKE) -C models

clean:
	$(MAKE) -C models clean

.PHONY: all models clean
