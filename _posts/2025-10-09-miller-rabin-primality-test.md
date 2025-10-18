---
title: Miller-Rabin primality test
author: sebastia
date: 2025-10-08 20:10:00 +0800
categories: [Cryptography]
tags: [cryptography, mathematics]
pin: true
toc: true
render_with_liquid: false
math: true
---

In the blog post - <a href="../prime-numbers">Prime Numbers</a> we have explained what is a primality test and specifically, the Miller-Rabin primality test. In this post we will implement a prototype of it in Python. Please, do not use it for production, it's just a dummy code to understand the Miller-Rabin algoirthm better. Before coding the final algorithm we need some other ingredients, we will explain from the most basic to the final algorithm.

## Exteneded Euclidean Algorithm

The greatest common divisor of two numbers $a$ and $b$ is the larger number that divides both $a$ and $b$ leaving a reminder of zero. For instance, what is the greatest common divisor of 10 and 45?. What we would do is to decompose in it's prime numbers both numbers: $10=5 \cdot 2$, $45=3 \cdot 5 \cdot 3$. The largest common divisor is $5$. The [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) can find the greatest common divisor of two numbers


$$
a=1067; b=462 \rightarrow 1067 = q_0 \times 462 +r_0 \rightarrow q_0=2; r_0=143
a=462; b=143 \rightarrow 462 = q_1 \times 143 + r_1 \rightarrow q_1=3; r_1=33
a=143; b=33 \rightarrow 143 = q_2 \times 33 + r_2 \rightarrow q_2=4; r_2=11
$$