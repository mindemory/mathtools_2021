function [v_x, v_y, perp_dist] = inner_prod(u_hat, v)
    % Given a vector v and a unit vector u_hat, this function computes:
    % i) v_x which is the component of v along u_hat. This can be computed
    % by taking the dot product of the two vectors.
    % ii) v_y which is the component of v along a unit vector orthogonal to
    % u_hat. Let this projection be w. Since v_x + w = v, we can compute
    % w = v - v_x
    % iii) The distance of v from the component v_x that lies along u_hat is
    % basically the length of v_y, which can be computed as the norm of v_y 
    v_x = u_hat'*v;
    v_y = v - v_x;
    perp_dist = norm(v_y, 2);
end