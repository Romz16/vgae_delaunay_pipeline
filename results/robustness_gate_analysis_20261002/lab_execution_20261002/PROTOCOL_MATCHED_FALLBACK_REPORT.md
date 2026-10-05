# Fallback equivalente para comparadores SOTA

Seleção refeita exclusivamente por macro-F1 de validação. Cada método recebeu o candidato `original`; o teste foi usado somente depois da seleção.

## Regra comum de fallback

O snapshot SDRF de 0% possui a topologia original, mas foi treinado em uma invocação separada e não reproduz bit a bit os valores do baseline salvo. Para evitar que qualquer método receba uma realização diferente do original, ele foi removido da grade SDRF e o mesmo baseline original salvo foi adicionado explicitamente a Proposed, SDRF e DiffWire CT.

## Resultados
- DiffWire CT: ganho médio contra o original = 0.520 p.p.; fallback para original = 85.1% (n=1080).
- Proposed: ganho médio contra o original = 4.364 p.p.; fallback para original = 36.4% (n=1080).
- SDRF: ganho médio contra o original = -0.292 p.p.; fallback para original = 42.7% (n=1080).

## Comparações pareadas com fallback em ambos os lados
- Proposed fallback minus SDRF fallback: 4.656 p.p. (IC 95% [1.529, 8.281], n=1080).
- Proposed fallback minus DiffWire CT fallback: 3.844 p.p. (IC 95% [0.975, 7.179], n=1080).
