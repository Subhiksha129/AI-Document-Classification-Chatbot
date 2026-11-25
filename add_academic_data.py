
import pandas as pd
import json
import os

print("Creating academic training data...", flush=True)

academic_data = [
    # Computer Science
    {
        "text": "Computer science is the study of computers, algorithms, programming languages and computational systems.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Computer Science"]
    },
    {
        "text": "Programming involves writing instructions using languages such as Python, Java and C++ to develop software applications.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Programming"]
    },

    # Data Structures and Algorithms
    {
        "text": "Data structures organize data efficiently. Arrays, linked lists, stacks, queues, trees and graphs are common data structures.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Data Structures and Algorithms"]
    },
    {
        "text": "Sorting algorithms include bubble sort, selection sort, insertion sort, merge sort and quick sort. Searching algorithms include linear search and binary search.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Data Structures and Algorithms"]
    },

    # Database Management
    {
        "text": "Database management systems store, organize and retrieve data. SQL is used to create tables, insert records and execute queries.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Database Management"]
    },
    {
        "text": "Database normalization reduces data redundancy. Primary keys, foreign keys and relationships maintain data integrity.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Database Management"]
    },

    # Operating Systems
    {
        "text": "An operating system manages computer hardware, memory, processes, files and input output devices.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Operating Systems"]
    },
    {
        "text": "Process scheduling, deadlocks, memory management and CPU scheduling are important operating system concepts.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Operating Systems"]
    },

    # Computer Networks
    {
        "text": "Computer networks allow devices to communicate and share information using protocols such as TCP IP, HTTP and DNS.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Computer Networks"]
    },
    {
        "text": "The OSI model contains seven layers: physical, data link, network, transport, session, presentation and application.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Computer Networks"]
    },

    # Cybersecurity
    {
        "text": "Cybersecurity protects computer systems, networks and data from unauthorized access, malware, phishing and cyber attacks.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Cybersecurity"]
    },
    {
        "text": "Encryption, authentication, firewalls and intrusion detection systems help protect information from security threats.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Cybersecurity"]
    },

    # Artificial Intelligence
    {
        "text": "Artificial intelligence enables machines to perform tasks that normally require human intelligence, including reasoning and problem solving.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Artificial Intelligence"]
    },
    {
        "text": "Machine learning is a branch of artificial intelligence that uses data and algorithms to identify patterns and make predictions.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Machine Learning"]
    },

    # Mathematics
    {
        "text": "Mathematics includes algebra, calculus, probability, statistics and discrete mathematics used in engineering and computer science.",
        "main_labels": ["Education"],
        "detail_labels": ["Mathematics"]
    },
    {
        "text": "Probability and statistics are used to analyze data, calculate distributions and understand uncertainty in mathematical problems.",
        "main_labels": ["Education"],
        "detail_labels": ["Mathematics"]
    },

    # Research
    {
        "text": "A research paper presents an investigation, methodology, experimental results, analysis and conclusions about a specific topic.",
        "main_labels": ["Education", "Technology"],
        "detail_labels": ["Research Papers"]
    },
    {
        "text": "Scientific research uses literature reviews, experiments, data collection and analysis to develop new knowledge.",
        "main_labels": ["Education"],
        "detail_labels": ["Research Papers"]
    }
]

new_df = pd.DataFrame(academic_data)

# Convert label lists into JSON strings
new_df["main_labels"] = new_df["main_labels"].apply(json.dumps)
new_df["detail_labels"] = new_df["detail_labels"].apply(json.dumps)

dataset_path = "dataset/processed_training_data.csv"

# Load existing dataset
old_df = pd.read_csv(dataset_path)

# Combine without changing the original file
combined_df = pd.concat(
    [old_df, new_df],
    ignore_index=True
)

output_path = "dataset/expanded_training_data.csv"

combined_df.to_csv(output_path, index=False)

print("Academic examples added:", len(new_df), flush=True)
print("Original records:", len(old_df), flush=True)
print("Expanded records:", len(combined_df), flush=True)
print("Saved to:", output_path, flush=True)
print("COMPLETED!", flush=True)
