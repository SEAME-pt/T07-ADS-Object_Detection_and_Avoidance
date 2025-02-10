#!/bin/bash

# =============================================================================
# Script for Updating Packages on Jetson Nano
# 
# Purpose:
# This script is designed to update the system packages on the Jetson Nano,
# excluding specific NVIDIA packages that are essential for the correct 
# operation of the system. These NVIDIA packages are held at their current 
# version to prevent compatibility issues during the upgrade process.
#
# Author: SEAME-PT/TEAM07
#
# Workflow:
# 1. Updates the package list from the repositories.
# 2. Holds back the essential NVIDIA packages from being upgraded.
# 3. Upgrades all packages listed in the PACKAGES variable, except for the 
#    NVIDIA packages that are held.
# =============================================================================

PACKAGES=(
    apparmor apport apport-gtk apt base-files bash bind9-host binutils binutils-aarch64-linux-gnu binutils-common bluez bluez-obexd ca-certificates
    chromium-browser chromium-browser-l10n chromium-codecs-ffmpeg-extra containerd cron dbus dbus-user-session dbus-x11 dirmngr distro-info-data
    dnsmasq-base docker.io dpkg dpkg-dev e2fsprogs ffmpeg fonts-opensymbol fwupd fwupd-signed ghostscript ghostscript-x gir1.2-notify-0.7 git git-man
    gnupg gnupg-l10n gnupg-utils gpg gpg-agent gpg-wks-client gpg-wks-server gpgconf gpgsm gpgv gstreamer1.0-gtk3 gstreamer1.0-plugins-good
    gstreamer1.0-pulseaudio gzip imagemagick imagemagick-6-common imagemagick-6.q16 iptables isc-dhcp-client isc-dhcp-common isc-dhcp-server keyutils
    klibc-utils krb5-locales libapparmor1 libapt-inst2.0 libapt-pkg5.0 libasn1-8-heimdal libavcodec57 libavdevice57 libavfilter6 libavformat57
    libavresample3 libavutil55 libbind9-160 libbinutils libbluetooth3 libc-bin libc-dev-bin libc6 libc6-dbg libc6-dev libcom-err2 libcups2 libcupsfilters1
    libcupsimage2 libcurl3-gnutls libcurl4 libdbus-1-3 libdns-export1100 libdns1100 libdpkg-perl libevdev2 libexempi3 libexpat1 libexpat1-dev libext2fs2
    libflac8 libfreerdp-client2-2 libfreerdp2-2 libfreetype6 libfribidi0 libfwupd2 libgmp10 libgnutls30 libgs9 libgs9-common libgssapi-krb5-2
    libgssapi3-heimdal libgstreamer-plugins-good1.0-0 libhcrypto4-heimdal libheimbase1-heimdal libheimntlm0-heimdal libhttp-daemon-perl libhx509-5-heimdal
    libice-dev libice6 libinput-bin libinput10 libip4tc0 libip6tc0 libiptc0 libirs-export160 libisc-export169 libisc169 libisccc160 libisccfg-export160
    libisccfg160 libjbig0 libjpeg-turbo-progs libjpeg-turbo8 libk5crypto3 libkeyutils1 libklibc libkpathsea6 libkrb5-26-heimdal libkrb5-3 libkrb5support0
    libksba8 libldap-2.4-2 libldap-common liblouis-data liblouis14 liblwres160 liblzma5 libmagickcore-6.q16-3 libmagickcore-6.q16-3-extra
    libmagickwand-6.q16-3 libmysqlclient20 libnautilus-extension1a libncurses5 libncursesw5 libnotify-bin libnotify4 libnss-myhostname libnss-systemd
    libnss3 libntfs-3g88 libnvinfer-dev libnvinfer8 libopenjp2-7 libpam-modules libpam-modules-bin libpam-runtime libpam-systemd libpam0g libpcre16-3
    libpcre3 libpcre3-dev libpcre32-3 libpcrecpp0v5 libperl5.26 libpixman-1-0 libpoppler-glib8 libpoppler73 libpostproc54 libprotobuf-lite10 libprotobuf10
    libpython2.7 libpython2.7-dev libpython2.7-minimal libpython2.7-stdlib libpython3.6 libpython3.6-minimal libpython3.6-stdlib librados2 libraw16
    librbd1 libreoffice-avmedia-backend-gstreamer libreoffice-base-core libreoffice-calc libreoffice-common libreoffice-core libreoffice-draw
    libreoffice-gnome libreoffice-gtk3 libreoffice-impress libreoffice-math libreoffice-ogltrans libreoffice-pdfimport libreoffice-style-breeze
    libreoffice-style-galaxy libreoffice-style-tango libreoffice-writer libroken18-heimdal libsasl2-2 libsasl2-modules libsasl2-modules-db libsdl1.2debian
    libsensors4 libsepol1 libsmbclient libsnmp-base libsnmp30 libspeex1 libspeexdsp1 libsqlite3-0 libss2 libssh2-1 libssl1.0.0 libssl1.1 libswresample2
    libswscale4 libsystemd0 libtiff5 libtinfo5 libudev1 libunwind8 libwayland-bin libwayland-client0 libwayland-cursor0 libwayland-dev libwayland-egl1
    libwayland-server0 libwbclient0 libwebp6 libwebpdemux2 libwebpmux3 libwind0-heimdal libwinpr2-2 libxml2 libxml2-dev libxpm4 libxslt1.1 libxtables12
    linux-firmware linux-libc-dev locales login multiarch-support nautilus nautilus-data ncurses-base ncurses-bin ncurses-term networkd-dispatcher ntfs-3g
    openssh-client openssh-server openssh-sftp-server openssl passwd perl perl-base perl-modules-5.26 poppler-utils python-apt-common python-pil
    python-pkg-resources python2.7 python2.7-dev python2.7-minimal python3-apport python3-apt python3-jwt python3-louis python3-mako python3-pkg-resources
    python3-problem-report python3-protobuf python3-uno python3.6 python3.6-minimal rsync rsyslog runc samba-libs snapd sudo systemd systemd-sysv tar
    thunderbird thunderbird-gnome-support tzdata udev uno-libs3 unzip update-notifier update-notifier-common ure vim vim-common vim-runtime wireless-regdb
    wpasupplicant xserver-common xserver-xephyr xserver-xorg-core xserver-xorg-legacy xwayland xxd xz-utils zlib1g zlib1g-dev
)

sudo apt update || { echo "Fail to update"; exit 1; }

# MAke sure that the following packages are held back
sudo apt-mark hold nvidia-l4t-bootloader nvidia-l4t-xusb-firmware nvidia-l4t-initrd


sudo apt install --only-upgrade "${PACKAGES[@]}"
