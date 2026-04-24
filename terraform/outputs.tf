# ── Active subscription (sanity check — always visible in apply output) ───────

output "active_subscription" {
  description = "Confirms which subscription resources are being deployed into"
  value       = "${data.azurerm_subscription.current.display_name} (${data.azurerm_subscription.current.subscription_id})"
}

# ── VM access ─────────────────────────────────────────────────────────────────

output "vm_private_ips" {
  description = "Private IP address for each VM (VMs have no public IP — use az run-command to interact)"
  value       = { for k, nic in azurerm_network_interface.vm : k => nic.private_ip_address }
}

# ── Storage ───────────────────────────────────────────────────────────────────

output "storage_account_name" {
  description = "Azure Storage Account name (needed for result download commands)"
  value       = azurerm_storage_account.main.name
}

# ── Quick-reference commands ──────────────────────────────────────────────────

output "check_status_commands" {
  description = "az run-command to check experiment status on each VM"
  value = {
    for k in keys(var.experiments) :
    k => "az vm run-command invoke -g ${var.resource_group_name} -n vm-${k} --command-id RunShellScript --scripts '/opt/stategen/status.sh' --query 'value[0].message' -o tsv"
  }
}

output "download_results_command" {
  description = "Run this on your laptop after experiments finish to pull all results locally"
  value       = <<-EOT
    az storage blob download-batch \
      --account-name ${azurerm_storage_account.main.name} \
      --source results \
      --destination ./results/ \
      --auth-mode login
  EOT
}

output "compare_command" {
  description = "Run this after downloading results to produce the cross-model comparison table"
  value       = <<-EOT
    python experiments/compare_models.py \
      ${join(" \\\n      ", [for k, v in var.experiments : "\"results/${k}:${v}\""])} \
      --output results/cross_model.json
  EOT
}

output "destroy_command" {
  description = "Delete all VMs and resources when done (stops billing immediately)"
  value       = "terraform destroy -auto-approve  # or: az group delete -n ${var.resource_group_name} --yes"
}
