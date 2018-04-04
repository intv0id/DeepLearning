#!/bin/sh

# install required dependancies
apt update && apt install -y vim git

# setup git
git config --global user.name "Clément Trassoudaine"
git config --global user.email trassoud@eurecom.fr

# save credentials for 2 hours
git config credential.helper 'cache --timeout=7200'