TEST_MYSQL_DSN ?= rag:rag@tcp(localhost:33306)/rag_test
TEST_QDRANT ?= localhost:56334
TEST_DOCLING ?= http://localhost:5001
COMPOSE = docker compose -f deploy/docker-compose.yml --env-file .env

.PHONY: build test test-deps test-deps-down test-e2e test-e2e-pdf up down logs

build:
	go build -o bin/ragd ./cmd/ragd

test:
	go test ./...

# Throwaway MySQL + Qdrant for integration tests (databases are wiped per run).
test-deps:
	docker run -d --rm --name rag-test-mysql -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=rag_test \
		-e MYSQL_USER=rag -e MYSQL_PASSWORD=rag -p 33306:3306 mysql:8.4
	docker run -d --rm --name rag-test-qdrant -p 56333:6333 -p 56334:6334 qdrant/qdrant:latest
	until docker exec rag-test-mysql mysqladmin ping -urag -prag --silent 2>/dev/null; do sleep 2; done
	docker exec rag-test-mysql mysql -uroot -proot -e "GRANT ALL ON *.* TO 'rag'@'%'"

test-deps-down:
	docker stop rag-test-mysql rag-test-qdrant

test-e2e:
	RAG_TEST_MYSQL_DSN='$(TEST_MYSQL_DSN)' RAG_TEST_QDRANT=$(TEST_QDRANT) RAG_TEST_DOCLING_URL=$(TEST_DOCLING) go test -count=1 ./...

# Only the end-to-end test, verbose (needs test-deps and a docling on TEST_DOCLING).
test-e2e-pdf:
	RAG_TEST_MYSQL_DSN='$(TEST_MYSQL_DSN)' RAG_TEST_QDRANT=$(TEST_QDRANT) RAG_TEST_DOCLING_URL=$(TEST_DOCLING) \
		go test -count=1 -v -run 'TestEndToEnd/pdf' ./internal/api/

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f ragd
