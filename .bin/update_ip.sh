# script to automatically update the security group to allow ssh access from the current IP address
aws ec2 authorize-security-group-ingress \
  --group-id sg-051a80de4d03118d1 \
  --protocol tcp \
  --port 22 \
  --cidr $(curl -s https://checkip.amazonaws.com)/32
