---
title: Initial Raspberry Pi configuration
author: sebastia
date: 2024-11-25 6:15:00 +0800
categories: [Raspberry, Linux]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

I'm a big fan of Raspberry Pi's (RP), so far I had three, the second, the third and the fourth generation. It all started as a side project to have fun: host services, setup an ssh server, install potsgresql and `SELECT DELETE database` without having an eery feeling while doing so. Basically mess around with a computer I didn't care about.

I found out that you can actually do many things in an RP: syncrhonizing files and notes (encrypted), download files through Torrent, setting up a small service to record cryptocurrency market fluctuation... All that in a low power consupmtion device!. I decided to document my raspberry setup right before I moved to California. Why then?. Well... I wanted a clean computer to SSH into and also host a VPN so that I could watch Spanish shows. Yes, I know... you expected a better answer... Two years ago I wrote all the process in my personal notes. Now it's time to publish those.

This first post is the basic setup, installing the raspberry pi operating system and the basic software.

## Format the SD card and install the OS

A raspberry pi has an SD card where the operating system is copied, which operating system should we install?. RPs have their own operating system based on Debian. Currenlty we have versions on Bookworm (version 12) or Bullseye (version 11), as they are the [latests versions](https://en.wikipedia.org/wiki/Debian_version_history) of Debian today. Check the  [RP OS downloads page](https://www.raspberrypi.com/software/operating-systems/) for the latests versions. How do we download and install any of these operating systems? Here is the easy way: 

* Plug the SD card into your laptop. If you don't have a slot for it, just search for "SD card adapter" and buy one.
* Download the [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
* Open the the image loader
  * Select your device (in my case Raspberry 4)
  * Choose Operating System. Normally 64-bit and lite. No need to install any desktop or extra apps.
  * Choose the storage: your SD card
  * At some point the app will ask you several things, if it doesn't make sure you find them before you flash the operating system. Click enable SSH and set up a password, this is important as we don't want to use a screen for first login to the RP. If you don't set up a password and leave it as default, the user is `pi` and the password `raspberry` (or that's what we used to have in old versions of the OS). Don't change the hostname unless you have more than one RP living in the same network. If you will use WiFi to connect your device set it up now in this section however it is always better to connect the RP to the router through Ethernet cable for speed reasons.

After this you should be all set, extract your SD card safely and plug it in to your RP.

## First Raspberry Pi boot and SSH

The next step is to connect your Raspberry Pi with the SD card to the power source and to your router using the Ethernet cable. Start the Raspberry Pi and do go your personal computer:

- Go to your router UI (in my case http://192.168.1.1) and identify the IP of your Raspberry. Let's say it is `192.168.1.128`.
- SSH to the raspberry `ssh pi@192.168.1.128` and type the password you inserted. Congrats, you are now logged in to your raspberry pi.

## Generating SSH key pair for login

Using password is not recommended, it is fine for the first boot but is not as secure as generating a pair of SSH keys. This pair consist of a private and a public key, the first is the one you keep in your laptop whereas the second is the one you share with the machine you want to connect to, in this case the RP. 

Let's generate a new SSH key pair with a more secure algorithm than [RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem)), I choose [ed25519](https://ed25519.cr.yp.to/).

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_raspberry -C "raspberry key"

# # if you still want to use RSA run at least 4096 bytes
# ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_raspberry -C "raspberry key"
```
Do not add a passkey when you are prompted, it will add complexity to your SSH. Without password the key is already quite secure. Now ls to your `~/.ssh` directory:

```bash
ls ~/.ssh
```
Two files should appear `id_raspberry` and `id_raspberry.pub`. We need to copy the latter to the raspberry, we can use the command `ssh-copy-id` for it (assuming `192.168.1.128` is the raspberry pi IP):

```bash
ssh-copy-id -i ~/.ssh/id_raspberry.pub pi@192.168.1.128
```

Now try to SSH to your raspberry using the user and the password as `ssh pi@192.168.1.128` and once logged in see that the key has been added in `~/.ssh/authorized_keys` by "catting" it `cat ~/.ssh/authorized_keys`. At this point you should be able to SSH without a password with the key as

```bash
ssh -i ~/.ssh/id_raspberry pi@192.168.1.128
```

## Easier SSH

The last command is not nice, you have to remember the identity file, the IP etc... I normally add a config file to simplify the login command. Create a file:

```bash
touch ~/.ssh/config
```

and copy the following in it

```bash
Host raspberry_local
HostName 192.168.1.128
User pi
Port 22
IdentityFile ~/.ssh/id_raspberry
```

Now you can login by simply typing in the command line `ssh raspberry_local`. The config above is very useful if you want to add other machines you normally SSH into. It is also convenient to create an alias, add in `~/.zshrc` (I use ZSH) or `~/.bashrc` (if bash):

```ssh
alias rpi_local='ssh raspberry_local'
```

Then `source ~/.zshrc` and ssh as `rpi_local` in the command line.


## Disable login with password

Once the passwordless SSH is enabled it is time to disable the password SSH, after all that's why we set up the ssh key pair. SSH to the raspberry

```bash
ssh raspberry_local
```

And install `vim`

``bash
sudo apt-get install vim
```

open the config file `vim /etc/ssh/sshd_config` and modify PasswordAuthentication to

```bash
PasswordAuthentication no
```

Now reboot the raspberry

```bash
sudo reboot
```

And wait for the machine to restart. It may take a minute or two this time. 

## Disable desktop

If you happened to install the desktop OS (sometimes the Lite version didn't work for me) it is time to disable it. It uses resources and after all we want to SSH only to the machine.

Once logged into the raspberry type `sudo raspi-config` to see the menu, then go to `System Options -> Boot/Auto Login` and select `Console`, then reboot with `sudo reboot`.

Finally ssh back to the raspberry and upgrade all the software in the system to have the latest versions on all packages

```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt full-upgrade
```

## Final notes

Now you have the basic raspberry setup, SSH using public-private key pair, disabled the desktop and with the most updated software that comes with the operating system. To me this is the basic start, a blank page. In the following posts we will configure internet access and create a VPN.

