terraform {
  required_version = ">= 1.3"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

# Guard: fail fast if Terraform is pointed at the wrong subscription.
# This prevents accidental deploys to Cloud Computing S26 or any other course account.
data "azurerm_subscription" "current" {}

resource "terraform_data" "subscription_guard" {
  lifecycle {
    precondition {
      condition     = data.azurerm_subscription.current.subscription_id == var.subscription_id
      error_message = <<-EOT
        ERROR: Active subscription is '${data.azurerm_subscription.current.display_name}' (${data.azurerm_subscription.current.subscription_id}).
        Expected 'Azure for Students' (${var.subscription_id}).
        Run: az account set --subscription "Azure for Students"
      EOT
    }
  }
}

# ── SSH key (auto-generated — you don't need to manage this manually) ──────────
# Saved to terraform/stategen_ssh.pem in case you ever need to SSH in for debug.

resource "tls_private_key" "ssh" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "local_sensitive_file" "ssh_private_key" {
  content         = tls_private_key.ssh.private_key_pem
  filename        = "${path.module}/stategen_ssh.pem"
  file_permission = "0600"
}

# ── Resource group ─────────────────────────────────────────────────────────────

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
}

# ── Storage: code upload + results collection ──────────────────────────────────

resource "azurerm_storage_account" "main" {
  # Storage account names must be globally unique, 3-24 chars, lowercase+digits only
  name                     = "stgexp${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "azurerm_storage_container" "code" {
  name                  = "code"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "results" {
  name                  = "results"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# Upload the zipped project from your local machine
resource "azurerm_storage_blob" "code" {
  name                   = "stategen.zip"
  storage_account_name   = azurerm_storage_account.main.name
  storage_container_name = azurerm_storage_container.code.name
  type                   = "Block"
  source                 = var.code_zip_path
}

# SAS token so VMs can download the zip without needing account keys
data "azurerm_storage_account_sas" "code" {
  connection_string = azurerm_storage_account.main.primary_connection_string
  https_only        = true

  resource_types {
    service   = false
    container = false
    object    = true
  }

  services {
    blob  = true
    queue = false
    table = false
    file  = false
  }

  start  = formatdate("YYYY-MM-DD", timestamp())
  expiry = formatdate("YYYY-MM-DD", timeadd(timestamp(), "${var.sas_validity_hours}h"))

  permissions {
    read    = true
    write   = false
    delete  = false
    list    = false
    add     = false
    create  = false
    update  = false
    process = false
    tag     = false
    filter  = false
  }
}

# SAS token for VMs to upload results back
data "azurerm_storage_account_sas" "results_write" {
  connection_string = azurerm_storage_account.main.primary_connection_string
  https_only        = true

  resource_types {
    service   = false
    container = true
    object    = true
  }

  services {
    blob  = true
    queue = false
    table = false
    file  = false
  }

  start  = formatdate("YYYY-MM-DD", timestamp())
  expiry = formatdate("YYYY-MM-DD", timeadd(timestamp(), "${var.sas_validity_hours}h"))

  permissions {
    read    = true
    write   = true
    delete  = false
    list    = true
    add     = true
    create  = true
    update  = true
    process = false
    tag     = false
    filter  = false
  }
}

locals {
  code_url       = "${azurerm_storage_blob.code.url}${data.azurerm_storage_account_sas.code.sas}"
  methods_str    = join(" ", var.methods)
  bcb_flag       = var.bcb_server_url != "" ? "--bcb_server ${var.bcb_server_url}" : ""
  active_api_key = var.google_api_key != "" ? var.google_api_key : var.cmu_api_key
}

# ── Networking (shared across all VMs) ────────────────────────────────────────

resource "azurerm_virtual_network" "main" {
  name                = "stategen-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

resource "azurerm_subnet" "main" {
  name                 = "stategen-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_network_security_group" "main" {
  name                = "stategen-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  # No inbound rules needed — VMs only make outbound API calls.
  # az vm run-command invoke goes through the Azure management plane, not SSH.
}

resource "azurerm_subnet_network_security_group_association" "main" {
  subnet_id                 = azurerm_subnet.main.id
  network_security_group_id = azurerm_network_security_group.main.id
}

# ── One NIC + VM per experiment (no public IP needed — using az run-command) ───
# Azure for Students blocks Basic SKU public IPs (limit 0).
# az vm run-command invoke works through the Azure management API without
# needing a public IP, so VMs only need a private IP on the subnet.

resource "azurerm_network_interface" "vm" {
  for_each            = var.experiments
  name                = "nic-${each.key}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.main.id
    private_ip_address_allocation = "Dynamic"
  }
}

resource "azurerm_linux_virtual_machine" "vm" {
  for_each            = var.experiments
  name                = "vm-${each.key}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  size                = var.vm_size
  admin_username      = var.admin_username

  # cloud-init runs automatically on first boot — sets up the experiment
  custom_data = base64encode(templatefile("${path.module}/templates/cloud_init.yaml.tpl", {
    code_url        = local.code_url
    model           = each.value
    experiment_name = each.key
    cmu_api_key     = local.active_api_key
    base_url        = var.base_url
    methods_str     = local.methods_str
    bcb_flag        = local.bcb_flag
    storage_account = azurerm_storage_account.main.name
    results_sas     = data.azurerm_storage_account_sas.results_write.sas
    max_new_tokens  = var.max_new_tokens
  }))

  network_interface_ids = [azurerm_network_interface.vm[each.key].id]

  admin_ssh_key {
    username   = var.admin_username
    public_key = tls_private_key.ssh.public_key_openssh
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
    disk_size_gb         = 32
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  tags = {
    experiment = each.key
    model      = each.value
    project    = "stategen"
  }
}
