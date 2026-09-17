/** Mulberry32 — deterministic, tiny, good enough for Monte Carlo demos. */
export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(min = 0, max = 1): number {
    return min + (max - min) * this.next();
  }

  int(min: number, max: number): number {
    return Math.floor(this.uniform(min, max + 1));
  }

  gaussian(mean = 0, std = 1): number {
    const u = Math.max(this.next(), 1e-12);
    const v = this.next();
    const mag = Math.sqrt(-2 * Math.log(u));
    return mean + std * mag * Math.cos(2 * Math.PI * v);
  }

  poisson(lambda: number): number {
    if (lambda <= 0) return 0;
    if (lambda < 40) {
      const L = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k += 1;
        p *= this.next();
      } while (p > L);
      return k - 1;
    }
    return Math.max(0, Math.round(this.gaussian(lambda, Math.sqrt(lambda))));
  }

  pick<T>(items: readonly T[]): T {
    return items[this.int(0, items.length - 1)]!;
  }
}

export function bitFlipF32(value: number, bit: number): number {
  const buf = new ArrayBuffer(4);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = value;
  u[0] ^= 1 << bit;
  return f[0];
}
