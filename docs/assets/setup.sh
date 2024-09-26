#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if a namespace exists
function ns_exists() {
  ip netns list | grep -w "$1" &> /dev/null
}

# Function to add a namespace if it does not exist
function create_namespace() {
  if ns_exists "$1"; then
    echo "Namespace $1 already exists."
  else
    ip netns add "$1"
    echo "Namespace $1 created."
  fi
}

# Function to create veth pairs and check for existence
function create_veth_pair() {
  if ip link show "$1" &> /dev/null; then
    echo "$1 already exists."
  else
    ip link add "$1" type veth peer name "$2"
    echo "Created veth pair: $1 <--> $2."
  fi
}

# Function to assign IP address to veth interface in a namespace
function assign_ip_in_ns() {
  if ip netns exec "$1" ip addr show "$2" | grep "$3" &> /dev/null; then
    echo "IP address $3 is already set for $2 in namespace $1."
  else
    ip netns exec "$1" ip addr add "$3" dev "$2"
    echo "Assigned IP address $3 to $2 in namespace $1."
  fi
}

# Function to manually add ARP entry and verify success
function add_arp_entry() {
  local ip_addr=$1
  local mac_addr=$2
  sudo arp -s "$ip_addr" "$mac_addr"
  if arp -n | grep "$ip_addr" &> /dev/null; then
    echo "Successfully added ARP entry for $ip_addr with MAC $mac_addr."
  else
    echo "Failed to add ARP entry for $ip_addr."
    exit 1
  fi
}

# Create namespaces for the load balancer and backends
create_namespace "lb"
create_namespace "h2"
create_namespace "h3"

# Create veth pairs for LB to Local Machine, and LB to backends
create_veth_pair "veth0" "veth0_lb"
create_veth_pair "veth2" "veth2_lb"
create_veth_pair "veth3" "veth3_lb"

# Attach veth pairs to their respective namespaces
ip link set veth0_lb netns lb
ip link set veth2 netns h2   # Attach veth2 to h2 namespace
ip link set veth3 netns h3   # Attach veth3 to h3 namespace
ip link set veth2_lb netns lb
ip link set veth3_lb netns lb

# Assign IP addresses with separate subnets to namespaces and host
assign_ip_in_ns "lb" "veth0_lb" "10.0.0.10/24"  # Load balancer subnet
assign_ip_in_ns "lb" "veth2_lb" "10.0.1.1/24"   # h2 subnet
assign_ip_in_ns "lb" "veth3_lb" "10.0.2.1/24"   # h3 subnet

assign_ip_in_ns "h2" "veth2" "10.0.1.2/24"      # h2 namespace IP
assign_ip_in_ns "h3" "veth3" "10.0.2.2/24"      # h3 namespace IP

# Assign IP for veth0 on local machine (direct connection to LB)
if ip addr show veth0 | grep "10.0.0.11" &> /dev/null; then
  echo "IP address 10.0.0.11 is already set for veth0."
else
  ip addr add 10.0.0.11/24 dev veth0
  echo "Assigned IP address 10.0.0.11 to veth0."
fi

# Bring up interfaces
ip link set veth0 up
ip netns exec lb ip link set veth0_lb up
ip netns exec lb ip link set veth2_lb up
ip netns exec lb ip link set veth3_lb up

# Enable IP forwarding in the load balancer namespace
ip netns exec lb sysctl -w net.ipv4.ip_forward=1

# Add ARP entries for load balancer connection
mac_lb_veth0=$(sudo ip netns exec lb ip link show veth0_lb | grep ether | awk '{print $2}')
sudo arp -s 10.0.0.10 $mac_lb_veth0
echo "Added ARP entry for 10.0.0.10 with MAC $mac_lb_veth0."

# Add static ARP entries for the backend namespaces
mac_h2=$(sudo ip netns exec lb ip link show veth2_lb | grep ether | awk '{print $2}')
sudo arp -s 10.0.1.2 $mac_h2
echo "Added ARP entry for 10.0.1.2 with MAC $mac_h2."

mac_h3=$(sudo ip netns exec lb ip link show veth3_lb | grep ether | awk '{print $2}')
sudo arp -s 10.0.2.2 $mac_h3
echo "Added ARP entry for 10.0.2.2 with MAC $mac_h3."

# Bring up interfaces in h2 and h3 namespaces
ip netns exec h2 ip link set dev veth2 up
ip netns exec h2 ip link set dev lo up
ip netns exec h3 ip link set dev veth3 up
ip netns exec h3 ip link set dev lo up

# Enable IP masquerading on the lb namespace (NAT)
ip netns exec lb iptables -t nat -A POSTROUTING -o veth2_lb -j MASQUERADE
ip netns exec lb iptables -t nat -A POSTROUTING -o veth3_lb -j MASQUERADE

# Disable reverse path filtering on veth0 and lb interfaces
sysctl -w net.ipv4.conf.veth0.rp_filter=0
ip netns exec lb sysctl -w net.ipv4.conf.veth0_lb.rp_filter=0
ip netns exec lb sysctl -w net.ipv4.conf.veth2_lb.rp_filter=0
ip netns exec lb sysctl -w net.ipv4.conf.veth3_lb.rp_filter=0

# Allow forwarding between veth0_lb and veth2_lb in lb namespace
ip netns exec lb iptables -A FORWARD -i veth0_lb -o veth2_lb -j ACCEPT
ip netns exec lb iptables -A FORWARD -i veth2_lb -o veth0_lb -j ACCEPT

# Allow forwarding between veth0_lb and veth3_lb in lb namespace
ip netns exec lb iptables -A FORWARD -i veth0_lb -o veth3_lb -j ACCEPT
ip netns exec lb iptables -A FORWARD -i veth3_lb -o veth0_lb -j ACCEPT

########################################################################
echo ""
echo "Start checking..."
echo ""

# Check ARP table on the host
echo "Checking ARP table on the host..."
arp -n

# Check the status of veth2 in the h2 namespace
echo "Checking interface veth2 inside h2 namespace..."
ip netns exec h2 ip addr show veth2
ip netns exec h2 ip link show veth2

# Check the status of the routing table on the host
echo "Checking routing table on the host..."
ip route show

# Check IP forwarding status on the host
echo "Checking IP forwarding status on the host..."
sysctl net.ipv4.ip_forward

# Check IP forwarding status in the load balancer namespace (lb)
echo "Checking IP forwarding status in lb namespace..."
ip netns exec lb sysctl net.ipv4.ip_forward

# Additional checks for troubleshooting
echo "Checking interface states inside lb namespace..."
ip netns exec lb ip link show

echo "Checking ARP table inside lb namespace..."
ip netns exec lb ip neigh show

echo "Checking routing table inside lb namespace..."
ip netns exec lb ip route show

echo "Checking packet statistics for veth interfaces in lb namespace..."
ip netns exec lb ip -s link show veth0_lb
ip netns exec lb ip -s link show veth2_lb
ip netns exec lb ip -s link show veth3_lb

echo "Checking NAT rules in lb namespace..."
ip netns exec lb iptables -t nat -L -v

echo "Checking reverse path filtering in h2 and h3..."
ip netns exec h2 sysctl net.ipv4.conf.all.rp_filter
ip netns exec h3 sysctl net.ipv4.conf.all.rp_filter

# Check if any firewall rules are blocking traffic
echo "Checking firewall rules on the host..."
sudo iptables -L
sudo iptables -t nat -L

# Debug routing table inside LB namespace
echo "LB namespace routing table:"
ip netns exec lb ip route

# Connectivity tests

# Function to test connectivity
function test_ping() {
  local target=$1
  local source=$2

  echo "Testing connectivity from $source to $target..."
  if ping -c 3 "$target" &> /dev/null; then
    echo "Ping from $source to $target succeeded."
  else
    echo "Ping from $source to $target failed."
    exit 1
  fi
}

# Test connectivity from host to load balancer (veth0 -> veth0_lb)
test_ping "10.0.0.10" "host"

# Test connectivity from host to h2 and h3
test_ping "10.0.1.2" "host"
test_ping "10.0.2.2" "host"

# Test connectivity within the load balancer namespace (lb -> h2 and lb -> h3)
echo "Testing connectivity from lb to h2..."
ip netns exec lb ping -c 3 10.0.1.2 || echo "Ping from lb to h2 failed."

echo "Testing connectivity from lb to h3..."
ip netns exec lb ping -c 3 10.0.2.2 || echo "Ping from lb to h3 failed."

echo "Network setup and tests completed!"
