DOCKER_COMPOSE_FILE = ./docker-compose.yml

build:
	docker compose -f $(DOCKER_COMPOSE_FILE) build

clean: down
	docker container prune --filter "until=$(shell date -u -d '1 minute ago' '+%Y-%m-%dT%H:%M:%SZ')" -f

cleanall: clean
	docker system prune --all --volumes --force

up:
	docker compose -f $(DOCKER_COMPOSE_FILE) up -d

down:
	docker compose -f $(DOCKER_COMPOSE_FILE) down --volumes --remove-orphans

logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f

status:
	docker ps -a

# Define a help target to display available Makefile targets
help:
	@echo "Available Makefile targets:"
	@echo "  status   : List the status of all containers"
	@echo "  build    : Build Docker images using Docker Compose"
	@echo "  up       : Run containers using Docker Compose in the background"
	@echo "  logs     : View logs of running containers"
	@echo "  down     : Stop and remove containers defined in the Docker Compose file"
	@echo "  clean    : Clean up containers and their networks and volumes"
	@echo "  cleanall : Clean up containers, also stopped, and remove unused images"
	@echo "  help     : Display this help message"
	@echo
	@echo "If needed, use 'docker exec -it <name_container> sh' to enter the container"
	@echo "Otherwise you can enter in a fresh container using 'docker run -it --rm <name_image> /bin/bash'"

# Set the default target when 'make' is called without any arguments
.DEFAULT_GOAL := help