package main

import (
	"fmt"

	"github.com/go-redis/redis/v8"
)

func searchQA(rdb *redis.Client, queryVector []float64) (string, error) {
	// Get all keys
	keys, err := rdb.Keys(ctx, "qa:*").Result()
	if err != nil {
		return "", err
	}

	// Find the most similar question
	var closestKey string
	var maxSimilarity float64 = -1

	for _, key := range keys {
		vectorKey := fmt.Sprintf("%s:vector", key)
		vector, err := getVector(rdb, vectorKey)
		if err != nil {
			return "", err
		}
		similarity := cosineSimilarity(queryVector, vector)
		if similarity > maxSimilarity {
			maxSimilarity = similarity
			closestKey = key
		}
	}

	// Get the answer for the closest question
	answer, err := rdb.HGet(ctx, closestKey, "answer").Result()
	if err != nil {
		return "", err
	}
	return answer, nil
}

func getVector(rdb *redis.Client, key string) ([]float64, error) {
	fields, err := rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, err
	}
	vector := make([]float64, len(fields))
	for i := 0; i < len(fields); i++ {
		val, ok := fields[fmt.Sprintf("dim%d", i)]
		if !ok {
			return nil, fmt.Errorf("missing dimension %d", i)
		}
		var fval float64
		fmt.Sscanf(val, "%f", &fval)
		vector[i] = fval
	}
	return vector, nil
}

func cosineSimilarity(vec1, vec2 []float64) float64 {
	dot := 0.0
	normVec1 := 0.0
	normVec2 := 0.0
	for i := range vec1 {
		dot += vec1[i] * vec2[i]
		normVec1 += vec1[i] * vec1[i]
		normVec2 += vec2[i] * vec2[i]
	}
	return dot / (math.Sqrt(normVec1) * math.Sqrt(normVec2))
}