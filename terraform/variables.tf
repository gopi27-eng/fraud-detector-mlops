variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Credit_card_fraud_detection"
  type        = string
  default     = "fraud-detector"
}

variable "cluster_name" {
  description = "Credit_Card_Fraud_Detection_Eks_cluster"
  type        = string
  default     = "fraud-detector-eks"
}