@doc doc"""
Minimise le problème : ``min_{||s||< \delta_{k}} q_k(s) = s^{t}g + (1/2)s^{t}Hs``
                        pour la ``k^{ème}`` itération de l'algorithme des régions de confiance

# Syntaxe
```julia
sk = Gradient_Conjugue_Tronque(fk,gradfk,hessfk,option)
```

# Entrées :   
   * **gradfk**           : (Array{Float,1}) le gradient de la fonction f appliqué au point xk
   * **hessfk**           : (Array{Float,2}) la Hessienne de la fonction f appliqué au point xk
   * **options**          : (Array{Float,1})
      - **delta**    : le rayon de la région de confiance
      - **max_iter** : le nombre maximal d'iterations
      - **tol**      : la tolérance pour la condition d'arrêt sur le gradient


# Sorties:
   * **s** : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \delta_{k}} q(s)``

# Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(gradfk, hessfk, options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        deltak = 2
        max_iter = 100
        tol = 1e-6
    else
        deltak = options[1]
        max_iter = options[2]
        tol = options[3]
    end

    n = length(gradfk)
    s = zeros(n)

    deltaj = deltak
    s_j = s
    g_j = gradfk
    p_j = -g_j
    H = hessfk  
    iter = 0
    continuer = true

    while continuer

        q(x) = 0.5 * transpose(x) * H * x + transpose(g_j) * x

        k_j = transpose(p_j) * H * p_j

        if k_j <= 0
            # On résoud le trinôme demandé (ax² + bx + c = 0)
            a = transpose(p_j) * p_j
            b = transpose(s_j) * p_j + transpose(p_j) * s_j
            c = transpose(s_j) * s_j - deltaj
            discriminant = b^2 - 4 * a * c

            if discriminant < 0
                println("Erreur, pas de solution réelle 1")
                break
            else
                x1 = (b - sqrt(discriminant)) / (2 * a)
                x2 = (b + sqrt(discriminant)) / (2 * a)
            end

            if q(s_j + x1 * p_j) <= q(s_j + x2 * p_j)
                sigma_j = x1
            else 
                sigma_j = x2

            end
            
            s = s_j + sigma_j * p_j
            break
        end

        alpha_j = transpose(g_j) * g_j / k_j

        if norm(s_j + alpha_j * p_j) >= deltaj


            # On résoud le trinôme demandé (ax² + bx + c = 0)
            a = transpose(p_j) * p_j
            b = transpose(s_j) * p_j + transpose(p_j) * s_j
            c = transpose(s_j) * s_j - deltaj
            discriminant = b^2 - 4 * a * c

            if discriminant < 0
                println("Erreur, pas de solution réelle 2")
                break
            else
                x1 = (b - sqrt(discriminant)) / (2 * a)
                x2 = (b + sqrt(discriminant)) / (2 * a)
            end

            if x1 * x2 >= 0
                println("Erreur, les solutions sont de même signe")
            else
                sigma_j = max(x1, x2)
            end
            
            s = s_j + sigma_j * p_j
            break
            
        end

        s_j_prec = s_j
        g_j_prec = g_j

        s_j = s_j + alpha_j * p_j
        g_j = g_j + alpha_j * H * p_j
        beta_j = (transpose(g_j) * g_j) / (transpose(g_j_prec) * g_j_prec)
        p_j = -g_j + beta_j * p_j



        iter = iter + 1

        # Nombre d'itération max atteint
        if iter >= max_iter
            s = s_j
            continuer = false
        end

        # Convergence atteinte
        if norm(s_j_prec - s_j) <= tol
            s = s_j
            continuer = false
        end
    end

    return s
end
