"""
baixar_dados_infodengue.py
Baixa os arquivos brutos grandes (não versionados) usados pelos notebooks do QAE:
    notebooks/data/dengue.csv.gz  (~23 MB)
    notebooks/data/climate.csv.gz (~281 MB)

Fonte: FTP público do InfoDengue (data_sprint_2025). Dados públicos e agregados.
Uso:  python scripts/baixar_dados_infodengue.py
"""
import ftplib
import hashlib
import os

DEST = os.path.join("notebooks", "data")
ARQUIVOS = ["dengue.csv.gz", "climate.csv.gz"]
FTP_HOST = "info.dengue.mat.br"
FTP_DIR = "/data_sprint_2025"


def sha256(caminho: str) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar(nome: str) -> str:
    os.makedirs(DEST, exist_ok=True)
    destino = os.path.join(DEST, nome)
    if os.path.exists(destino):
        print(f"[ok] {nome} já existe — pulando download.")
    else:
        print(f"[..] baixando {nome} de ftp://{FTP_HOST}{FTP_DIR}/ ...")
        with ftplib.FTP(FTP_HOST) as ftp:
            ftp.login("anonymous", "anonymous@domain.com")
            ftp.cwd(FTP_DIR)
            with open(destino, "wb") as f:
                ftp.retrbinary(f"RETR {nome}", f.write)
        print(f"[ok] {nome} baixado.")
    print(f"     SHA-256: {sha256(destino)}")
    return destino


if __name__ == "__main__":
    for arq in ARQUIVOS:
        baixar(arq)
    print("\nConcluído. Registre os checksums acima na seção 'Disponibilidade de "
          "dados' do relatório para garantir a integridade do snapshot.")
