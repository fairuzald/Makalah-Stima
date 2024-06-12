CREATE TABLE Menu (
    idMenu VARCHAR(36) PRIMARY KEY DEFAULT UUID(),
    nama VARCHAR(255) NOT NULL,
    harga INT NOT NULL CHECK (harga >= 0),
    rating FLOAT CHECK (rating >= 0 AND rating <= 5),
    deskripsi VARCHAR(255) DEFAULT 'Deskripsi Menu Belum Tersedia',
    tipe ENUM ('Makanan', 'Minuman') NOT NULL
);

CREATE TABLE PenyediaBahan (
    email VARCHAR(255) PRIMARY KEY CHECK (email REGEXP '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    nama VARCHAR(255) NOT NULL,
    nomorTelepon VARCHAR(13) CHECK (nomorTelepon REGEXP '^\\+?[0-9]+$' AND LENGTH(nomorTelepon) <= 13)
);
-- 
CREATE TABLE Bahan (
    idBahan VARCHAR(36) PRIMARY KEY DEFAULT UUID(),
    nama VARCHAR(255) NOT NULL,
    stok INT DEFAULT 0 CHECK (stok >= 0)
);

CREATE TABLE Pegawai (
    NIKPegawai VARCHAR(255) PRIMARY KEY CHECK (LENGTH(NIKPegawai) = 16),
    nama VARCHAR(255) NOT NULL,
    alamat VARCHAR(255),
    nomorTelepon VARCHAR(13) CHECK (nomorTelepon REGEXP '^\\+?[0-9]+$' AND LENGTH(nomorTelepon) <= 13)
);

CREATE TABLE Pengunjung (
    email VARCHAR(255) PRIMARY KEY CHECK (email REGEXP '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    nama VARCHAR(255) NOT NULL,
    nomorTelepon VARCHAR(13) CHECK (nomorTelepon REGEXP '^\\+?[0-9]+$' AND LENGTH(nomorTelepon) <= 13)
);

CREATE TABLE Transaksi (
    nomorTransaksi VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    metodePembayaran ENUM ('Tunai', 'Kartu Kredit', 'QRIS', 'Debit') NOT NULL, 
    totalHarga INT NOT NULL CHECK (totalHarga >= 0),
    tanggalTransaksi DATETIME DEFAULT CURRENT_TIMESTAMP, 
    emailPengunjung VARCHAR(255), 
    NIKPegawai VARCHAR(255),
    FOREIGN KEY (emailPengunjung) REFERENCES Pengunjung(email) ON DELETE SET NULL ON UPDATE CASCADE,
    FOREIGN KEY (NIKPegawai) REFERENCES Pegawai(NIKPegawai) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE TABLE Makanan (
    idMenu VARCHAR(36) PRIMARY KEY,
    tingkatKepedasan INT DEFAULT 0 CHECK (tingkatKepedasan >= 0 AND tingkatKepedasan <= 5),
    FOREIGN KEY (idMenu) REFERENCES Menu(idMenu) ON DELETE CASCADE ON UPDATE CASCADE 
);

CREATE TABLE Minuman (
    idMenu VARCHAR(36) PRIMARY KEY,
    isDingin BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (idMenu) REFERENCES Menu(idMenu) ON UPDATE CASCADE
);

CREATE TABLE BahanMenu (
    idMenu VARCHAR(36),
    idBahan VARCHAR(36),
    jumlah INT NOT NULL CHECK (jumlah >= 0),
    PRIMARY KEY (idMenu, idBahan),
    FOREIGN KEY (idMenu) REFERENCES Menu(idMenu) ON DELETE CASCADE ON UPDATE CASCADE ,
    FOREIGN KEY (idBahan) REFERENCES Bahan(idBahan) ON DELETE CASCADE ON UPDATE CASCADE 
);

CREATE TABLE PembelianBahan (
    emailPenyedia VARCHAR(255),
    idBahan VARCHAR(36),
    harga INT NOT NULL CHECK (harga >= 0),
    jumlahPembelian INT NOT NULL CHECK (jumlahPembelian >= 0),
    tanggalPembelian DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (emailPenyedia, idBahan, tanggalPembelian),
    FOREIGN KEY (emailPenyedia) REFERENCES PenyediaBahan(email) ON DELETE CASCADE ON UPDATE CASCADE ,
    FOREIGN KEY (idBahan) REFERENCES Bahan(idBahan) ON DELETE CASCADE ON UPDATE CASCADE 
);

CREATE TABLE DetailTransaksi (
    nomorTransaksi VARCHAR(36) NOT NULL,
    idDetailTransaksi VARCHAR(36) NOT NULL DEFAULT UUID() PRIMARY KEY,
    kuantitas INT NOT NULL CHECK (kuantitas >= 0),
    idMenu VARCHAR(36),
    FOREIGN KEY (nomorTransaksi) REFERENCES Transaksi(nomorTransaksi) ON DELETE CASCADE ON UPDATE CASCADE ,
    FOREIGN KEY (idMenu) REFERENCES Menu(idMenu) ON DELETE CASCADE ON UPDATE CASCADE 
);

CREATE TABLE Feedback (
    idFeedback VARCHAR(36) PRIMARY KEY,
    waktuFeedback DATETIME DEFAULT CURRENT_TIMESTAMP,
    ratingPelayanan INT CHECK (ratingPelayanan >= 0 AND ratingPelayanan <= 5),
    ratingKebersihan INT CHECK (ratingKebersihan >= 0 AND ratingKebersihan <= 5),
    komentar VARCHAR(255),
    ratingMenuOverall FLOAT CHECK (ratingMenuOverall >= 0 AND ratingMenuOverall <= 5),
    CONSTRAINT chk_feedback CHECK (NOT (ratingPelayanan = 5 AND ratingKebersihan = 0)),
    FOREIGN KEY (idFeedback) REFERENCES Transaksi(nomorTransaksi) ON DELETE CASCADE ON UPDATE CASCADE
);


CREATE TABLE RatingMenu (
    idDetailTransaksi VARCHAR(36) NOT NULL,
    idFeedback VARCHAR(36) NOT NULL,
    rating INT CHECK (rating >= 0 AND rating <= 5),
    PRIMARY KEY (idDetailTransaksi, idFeedback),
    FOREIGN KEY (idDetailTransaksi) REFERENCES DetailTransaksi(idDetailTransaksi) ON DELETE CASCADE ON UPDATE CASCADE ,
    FOREIGN KEY (idFeedback) REFERENCES Feedback(idFeedback) ON DELETE CASCADE ON UPDATE CASCADE 
);