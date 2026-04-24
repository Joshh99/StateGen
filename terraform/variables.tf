variable "subscription_id" {
  description = <<-EOT
    Azure subscription ID to deploy into.
    Pinned to 'Azure for Students' to avoid accidental charges on other course subscriptions.
    Do not change this unless you know what you are doing.
  EOT
  type    = string
  default = "b31eda3d-8b42-46e9-9b44-f6327d69e8c5"  # Azure for Students
}

variable "resource_group_name" {
  description = "Azure resource group for all experiment resources"
  type        = string
  default     = "stategen-experiments"
}

variable "location" {
  description = <<-EOT
    Azure region. Azure for Students is restricted to these 5 regions only:
      centralindia   — Central India
      italynorth     — Italy North
      spaincentral   — Spain Central
      austriaeast    — Austria East
      uaenorth       — UAE North
    VM location doesn't affect experiment speed — the bottleneck is LLM API
    response time (~5-30s), not the ~100ms extra network hop to the CMU gateway.
  EOT
  type    = string
  default = "italynorth"

  validation {
    condition     = contains(["centralindia", "italynorth", "spaincentral", "austriaeast", "uaenorth"], var.location)
    error_message = "Azure for Students only allows: centralindia, italynorth, spaincentral, austriaeast, uaenorth."
  }
}

variable "vm_size" {
  description = <<-EOT
    VM size. Azure for Students allows 4 standardBSFamily vCPUs in Italy North.
    Standard_B1ms (1 vCPU, 2GB RAM) lets us run 4 VMs within that quota.
    The bottleneck is LLM API latency (~5-30s/call), not CPU — B1ms is plenty.
  EOT
  type    = string
  default = "Standard_B1ms"
}

variable "admin_username" {
  description = "Linux admin username on each VM"
  type        = string
  default     = "stategen"
}

variable "cmu_api_key" {
  description = "CMU Andrew gateway API key (kept sensitive, never logged)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "google_api_key" {
  description = "Google AI API key for Gemini models via generativelanguage.googleapis.com"
  type        = string
  sensitive   = true
  default     = ""
}

variable "code_zip_path" {
  description = "Local path to the zipped project (created by the package.sh script)"
  type        = string
  default     = "../stategen_code.zip"
}

variable "bcb_server_url" {
  description = "Optional BigCodeBench server URL for real-time eval. Leave empty to use quick_check proxy."
  type        = string
  default     = ""
}

variable "experiments" {
  description = <<-EOT
    Map of experiment_name => model_id.
    experiment_name becomes the --experiment flag and the results subdirectory.
    model_id is passed as --model to run_bigcodebench.py.
  EOT
  type        = map(string)
  default = {
    gpt4omini    = "gpt-4o-mini"
    gpt4o        = "gpt-4o"
    claude       = "claude-sonnet-4-6"
    gemini       = "gemini-2.5-pro"
  }
}

variable "methods" {
  description = "Baselines to run on each VM. 'all' runs all 7 methods."
  type        = list(string)
  default     = ["all"]
}

variable "base_url" {
  description = "OpenAI-compatible base URL for all models"
  type        = string
  default     = "https://ai-gateway.andrew.cmu.edu/"
}

variable "sas_validity_hours" {
  description = "Hours the blob SAS token stays valid (must cover experiment runtime + buffer)"
  type        = number
  default     = 96
}

variable "max_new_tokens" {
  description = <<-EOT
    Max output tokens per LLM call. Raise to 16384+ for thinking models like gemini-2.5-pro
    (which consume most of the budget for chain-of-thought reasoning before producing output).
    Default 2048 is fine for DeepSeek, GPT-4o, and Claude.
  EOT
  type    = number
  default = 2048
}
