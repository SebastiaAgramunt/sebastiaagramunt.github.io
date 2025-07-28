---
title: Veracrypt
author: sebastia
date: 2025-07-27 11:42:00 +0800
categories: [Tools]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

[Veracrypt](https://veracrypt.io/en/Home.html) is a free, open-source encryption software used to:

* Create encrypted volumes (containers) to securely store files.
* Encrypt entire disks or partitions, including system drives.
* Protect sensitive data with strong encryption algorithms like AES, Serpent, and Twofish.
* Support hidden volumes, adding plausible deniability.

It's commonly used for securing data on laptops, USB drives, or external disks. I personally use it to encrypt my backups before saving them to an external cloud or physical devices. In this post we show how to install and use veracrypt in command line.

## TLDR

Setting up: Create a directory to store the `hc` files and a `keyfile.bin` file

```bash
# create dir
VERACRYPT_STORE=~/.VeracryptVolumes/
mkdir -p ${VERACRYPT_STORE}

# create keyfile.bin random file
KEYFILES=${HOME}/.ssh/keyfile.bin
dd if=/dev/urandom of=${KEYFILES} bs=512 count=1

# visually check the file
cat ~/.ssh/keyfile.bin |  hexdump -C
```

For interactive session (dynamically define every parameter) run `veracrypt -t -c` to create a volume, otherwise define constants to create the volume

```bash
SIZE="20MiB"
ENCRYPTION="AES"
HASH="SHA-512"
FILESYSTEM="exFAT"
PIM=120
VOLUME_NAME="secret_volume"
```

Now create volume (use a long and secure password along with the keyfile for better security)

```bash
veracrypt --text --create \
--size=${SIZE} \
--volume-type=normal \
--encryption=${ENCRYPTION} \
--hash=${HASH} \
--filesystem=${FILESYSTEM} \
--pim=${PIM} \
--keyfiles=${KEYFILES} \
--random-source /dev/urandom \
${VERACRYPT_STORE}/${VOLUME_NAME}.hc
```

Important note: It is best practice to not set the `--random-source` and be prompted to type 320 characters to generate the entropy. We just use the program `/dev/urandom` here because is more practical.

Mount the volume

```bash
veracrypt --text --mount \
${VERACRYPT_STORE}/${VOLUME_NAME}.hc \
--pim=${PIM} \
--protect-hidden=no \
--keyfiles=${KEYFILES} \
/Volumes/${VOLUME_NAME}
```

Add a file to your volume as an example

```bash
echo "Hey, this file is going to be encrypted" > /Volumes/${VOLUME_NAME}/encrypted_file.txt
```

And unmount the volume with


```bash
veracrypt --text --unmount ${VERACRYPT_STORE}/${VOLUME_NAME}.hc
```

## Install VeraCrypt

In MacOS just use brew

```bash
brew install --cask veracrypt 
```

for other OSs, please download from the [downloads page](https://veracrypt.io/en/Downloads.html) and install manually. Once installed type


```bash
veracrypt --text --help
```

to see the options. The `text` flag indicates command line. If you wish to open the UI, just type `veracrypt`. If you are on a Mac, `macFUSE` is also needed. Go to [macfuse releases](https://github.com/macfuse/macfuse/releases) page and donwload and install the latest stable release. As of today it is [macFUSE 4.10.2](https://github.com/macfuse/macfuse/releases/tag/macfuse-4.10.2). After installation you may need to restart your mac.

## CLI usage

In this section we won't explain the UI usage, for that you have a very nice [beginner's tutorial](https://veracrypt.io/en/Beginner%27s%20Tutorial.html) from Veracrypt. 

### Create a volume

In veracrypt, a volume is a virtual encrypted disk. It behaves like a real disk once mounted, but all data stored on it is automatically encrypted.

There are two main types of veracrypt volumes:

A file container is a single encrypted file that acts like a virtual drive. You mount it with veracrypt, and it appears as a new drive. Inside, you can store files and folders just like on a regular disk.

A partition or disk encryption in veracrypt encrypts an entire partition or physical disk (e.g., USB, external drive, or system drive). The whole drive is protected, and access requires a password at boot or when mounting.

Normally I work on the former, file containers. Let's crate one but first print on screen the options:

```bash
veracrypt --text --create --help
```

The options I use

```
--size=SIZE[K|KiB|M|MiB|G|GiB|T|TiB] or --size=max
 Use specified size when creating a new volume. If no suffix is indicated,
 then SIZE is interpreted in bytes. Suffixes K, M, G or T can be used to
 indicate a value in KiB, MiB, GiB or TiB respectively.
 If max is specified, the new volume will use all available free disk space.

--volume-type=TYPE
 Use specified volume type when creating a new volume. TYPE can be 'normal'
 or 'hidden'. See option -c for more information on creating hidden volumes.

 --encryption=ENCRYPTION_ALGORITHM
 Use specified encryption algorithm when creating a new volume. When cascading
 algorithms, they must be separated by a dash. For example: AES-Twofish.

 --hash=HASH
 Use specified hash algorithm when creating a new volume or changing password
 and/or keyfiles. This option also specifies the mixing PRF of the random
 number generator.

 --filesystem=TYPE
 Filesystem type to mount. The TYPE argument is passed to mount(8) command
 with option -t. Default type is 'auto'. When creating a new volume, this
 option specifies the filesystem to be created on the new volume.
 Filesystem type 'none' disables mounting or creating a filesystem.

 --pim=PIM
 Use specified PIM to mount/open a volume. Note that passing a PIM on the
 command line is potentially insecure as the PIM may be visible in the process
 list (see ps(1)) and/or stored in a command history file or system logs.

 -k, --keyfiles=KEYFILE1[,KEYFILE2,KEYFILE3,...]
 Use specified keyfiles when mounting a volume or when changing password
 and/or keyfiles. When a directory is specified, all files inside it will be
 used (non-recursively). Multiple keyfiles must be separated by comma.
 Use double comma (,,) to specify a comma contained in keyfile's name.
 Keyfile stored on a security token must be specified as
 token://slot/SLOT_NUMBER/file/FILENAME for a security token keyfile
 and emv://slot/SLOT_NUMBER for an EMV token keyfile.
 An empty keyfile (-k "") disables
 interactive requests for keyfiles. See also options --import-token-keyfiles,
 --list-token-keyfiles, --list-securitytoken-keyfiles, --list-emvtoken-keyfiles,
 --new-keyfiles, --protection-keyfiles.

 --random-source=FILE
 Use FILE as a source of random data (e.g., when creating a volume) instead
 of requiring the user to type random characters.
```

If you want to dynamically create the volume seeing the options on screen run `veracrypt -t -c`, sometimes it is easier to do this without predefining any configuration. In the following sections I define some convenient variables to define the volume, after all if we use this in a bash script we don't want to be prompted too much. Trying to automate as much as I can here.

#### Create a password protected volume

And my command to crate a volume of `20MB` in a file named `my_first_volume.hc` in the newly created `~/.VeracryptVolumes` directory:

```bash
mkdir ~/.VeracryptVolumes/
veracrypt --text --create \
--size=20MiB \
--volume-type=normal \
--encryption=AES \
--hash=SHA-512 \
--filesystem=exFAT \
--pim=120 \
--keyfiles="" \
--random-source /dev/urandom \
~/.VeracryptVolumes/my_first_volume.hc
```

Let's inspect this call

* Encryption is `AES` algorithm, see [encryption algorithms available](https://veracrypt.io/en/Encryption%20Algorithms.html).
* `exFAT` filesystem for maximum compatibility with Windows, MacOS and modern Linux distributions.
* `PIM` (Personal Iterations Multiplier) of 120. PIM controls the number of hash iterations used during the password derivation process when mounting a volume, the higher the more secure but also will take more time to mount the volume.
* `keyfiles` empty if you just want password protection. Add a random file for better protection
* `random-source` is set to `/dev/urandom`, a computer pseudo-random number generator (see an output of 100 random bytes in terminal with `head -c 100 /dev/urandom | hexdump -C`). Normally you would not introduce this parameter and would be expected to type 320 random characters at the moment of volume creation to increase entropy in the encryption.
* The last parameter is the name of the volume created

Once the above command is executed it will prompt to introduce your desired password twice and then will create the volume. Make sure you use strong passwords (15 characters minimum combining caps, numbers and non-ascii characters), a good page to generate those is [https://www.strongpasswordgenerator.org/](https://www.strongpasswordgenerator.org/). After successful execution of the command check the file has been created by executing `ls -lhat ~/.VeracryptVolumes`, getting something like:

```bash
-rw-------    1 sebas  staff    20M 27 Jul 13:32 my_first_volume.hc
```


### Create a password and key protected volume

A more secure way to create a volume is using two factor autenticaction. A `keyfile` is just a random file, a picture, a completely random file with characters etc. They are simply files whose contents are mixed with your password to derive the final encryption key. I chose to create a random file of 512 Bytes first with:

```bash
dd if=/dev/urandom of=$HOME/.ssh/keyfile.bin bs=512 count=1
```

whose content in hexadecimal can be checked with `cat ~/.ssh/keyfile.bin |  hexdump -C`. Now use the keyfile key and password to create the encrypted volume:

```bash
veracrypt --text --create \
--size=20MiB \
--volume-type=normal \
--encryption=AES \
--hash=SHA-512 \
--filesystem=FAT \
--pim=120 \
--random-source /dev/urandom \
--keyfiles="${HOME}/.ssh/keyfile.bin" \
~/.VeracryptVolumes/my_second_volume.hc
```

Keep the `keyfile` safe, because it is needed to mount the volume, without it you would have lost access to your encrypted data.


### Mount & unmount a volume

Mounting a volume is making it accessible in the filesystem. The same way you mount a USB drive in linux when you connect a USB you can mount a veracrypt volume. See the options with

```bash
veracrypt --text --mount --help
```

To unmount

```bash
veracrypt --text --unmount --help
```

In the next subsections we will mount and unmount the drives we created before.

#### Mount & unmount a volume with password

Once the file is created we use veracrypt to decrypt the volume and mount it in our filesystem. That way we can start saving files in the volume. Let's mount the two recently created volumes, 

```bash
veracrypt --text --mount \
~/.VeracryptVolumes/my_first_volume.hc \
--pim=120 \
--protect-hidden=no \
--keyfiles=""  \
/Volumes/my_first_volume
```

And the volume will be mounted in `/Volumes/my_first_volume` so to see the contents excuete `ls -lhat /Volumes/my_first_volume`. Or run

```bash
diskutil list
```
with a result like:

```
...
/dev/disk2 (disk image):
   #:                       TYPE NAME                    SIZE       IDENTIFIER
   0:                                                   +20.7 MB    disk2
```

Check in your Findr in MacOS that the volume is there under `my_first_volume` (that's possible because we use `exFAT`, if we used `FAT` it would appear as `NO NAME`) and you can start moving data to your volume!. When you finish adding the data just unmount the volume with:


```bash
veracrypt --text --unmount ~/.VeracryptVolumes/my_first_volume.hc
```

### Mount & unmount a volume with password and a keyfile

Similarly with the volume we created with the `keyfile` we just need to run:

```bash
veracrypt --text --mount \
~/.VeracryptVolumes/my_second_volume.hc \
--pim=120 \
--protect-hidden=no \
--keyfiles="${HOME}/.ssh/keyfile.bin" \
/Volumes/my_second_volume
```
and unmount with

```bash
veracrypt --text --unmount ~/.VeracryptVolumes/my_second_volume.hc
```

## Backup and save your volumes

Now you have your encrypted volumes with your encrypted files inside. That's great, but you still need to keep this secure. 

* Keep both, your encryption password and your keyfiles in a password manager like LastPass, KeePass (free and open source), Bitwarden, 1Password, ProtonPass, MegaPass... Any of these work.
* Copy the encrypted volumes in physical HD drives and cloud services like Google Drive. It's safe as they are encrypted so these providers won't be able to see the content. 

