// Install the official, checksum-pinned binary for local use and Vercel builds.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const version = '0.27.16';
const platforms = {
  win32: ['x86_64-pc-windows-msvc.zip', '480ec45490ec671f3ffdd5db3976ddb3967c073fe7128b654672ba0c2b1ecbb9'],
  linux: ['x86_64-unknown-linux-gnu.tar.gz', '62eed34f0c90a24b696b1969c8aad4340e11ec7264e1cf6fc375ad15c1db7663'],
};
const release = platforms[process.platform];
if (!release || process.arch !== 'x64') throw new Error('This installer supports Windows/Linux x64. Set AUDIOMARK_C2PATOOL_PATH for other platforms.');
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const destination = join(root, 'runtime-bin');
const temporary = mkdtempSync(join(tmpdir(), 'audiomark-c2patool-'));
try {
  const asset = `c2patool-v${version}-${release[0]}`;
  const response = await fetch(`https://github.com/contentauth/c2pa-rs/releases/download/c2patool-v${version}/${asset}`);
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);
  const archive = join(temporary, asset);
  writeFileSync(archive, Buffer.from(await response.arrayBuffer()));
  if (createHash('sha256').update(readFileSync(archive)).digest('hex') !== release[1]) throw new Error('c2patool archive checksum mismatch');
  execFileSync('tar', ['-xf', archive, '-C', temporary]);
  const executable = process.platform === 'win32' ? 'c2patool.exe' : 'c2patool';
  mkdirSync(destination, { recursive: true });
  const installed = join(destination, executable);
  copyFileSync(join(temporary, 'c2patool', executable), installed);
  chmodSync(installed, 0o755);
  console.log(execFileSync(installed, ['--version'], { encoding: 'utf8' }).trim());
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
