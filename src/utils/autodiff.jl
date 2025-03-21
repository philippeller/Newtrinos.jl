using ForwardDiff
import ForwardDiff: gradient

function gradient(f, p::NamedTuple)

    pvals = collect(values(p))
    pnames = Tuple(keys(p))
    
    function wrapped_f(x)
        p = NamedTuple{pnames}(x)
        f(p)
    end

    g = ForwardDiff.gradient(wrapped_f, pvals)

    return NamedTuple{pnames}(g)
        
end