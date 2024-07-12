package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/go-redis/redis/v8"
	"github.com/ynqa/go-nlp/embedding"
	"github.com/ynqa/go-nlp/embedding/glove"
)

var ctx = context.Background()

type QA struct {
	Question string
	Answer   string
	Vector   []float64
}

func main() {
	rdb := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
	})

	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		log.Fatalf("Could not connect to Redis: %v", err)
	}

	// Load GloVe embeddings
	emb, err := glove.Load("path/to/glove.6B.50d.txt")
	if err != nil {
		log.Fatalf("Could not load GloVe embeddings: %v", err)
	}

	qas := []QA{
		{"What is CCIE?", "CCIE is Cisco Certified Internetwork Expert.", nil},
		{"What is AWS?", "AWS is Amazon Web Services.", nil},
		{"What is Go?", "Go is a statically typed, compiled programming language designed at Google.", nil},
		{"What is Redis?", "Redis is an open-source, in-memory data structure store, used as a database, cache, and message broker.", nil},
		{"What is Kubernetes?", "Kubernetes is an open-source system for automating the deployment, scaling, and management of containerized applications.", nil},
	}

	// Vectorize and store questions and answers
	for i, qa := range qas {
		vector := vectorize(emb, qa.Question)
		qas[i].Vector = vector
		err := storeQA(rdb, qa)
		if err != nil {
			log.Fatalf("Could not store QA: %v", err)
		}
	}

	// Perform a search for a question
	question := "What is CCIE?"
	queryVector := vectorize(emb, question)
	answer, err := searchQA(rdb, queryVector)
	if err != nil {
		log.Fatalf("Could not search QA: %v", err)
	}
	fmt.Println("Answer:", answer)
}

func vectorize(emb embedding.Embedder, text string) []float64 {
	words := strings.Fields(text)
	vector := make([]float64, emb.Dim())
	for _, word := range words {
		vec, err := emb.Vector(word)
		if err == nil {
			for i, val := range vec {
				vector[i] += val
			}
		}
	}
	for i := range vector {
		vector[i] /= float64(len(words))
	}
	return vector
}

func storeQA(rdb *redis.Client, qa QA) error {
	key := fmt.Sprintf("qa:%s", qa.Question)
	err := rdb.HSet(ctx, key, "answer", qa.Answer).Err()
	if err != nil {
		return err
	}
	for i, val := range qa.Vector {
		err := rdb.HSet(ctx, fmt.Sprintf("%s:vector", key), fmt.Sprintf("dim%d", i), val).Err()
		if err != nil {
			return err
		}
	}
	return nil
}
