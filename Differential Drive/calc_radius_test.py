def calculate_radius(velocity_left, velocity_right):
    """ Calculate radius with given left and right velocity values.
    Parameters:
        velocity left: range [-1,1], (implement assert for this part)
        velocity right: range [-1,1]    
    Return:
        radius: radius of the circle
    """
    assert velocity_left>=-1 and velocity_left<=1, "The range of velocity is [-1,1]"
    assert velocity_right>=-1 and velocity_right<=1, "The range of velocity is [-1,1]"
    # distance between centers of two wheels
    WHEEL_DIST = 0.102
    
    ############################
    ### TODO: YOUR CODE HERE ###
    if velocity_right - velocity_left == 0:
      return 0 # return 0 when vl == vr
    radius = (WHEEL_DIST / 2) * (velocity_left + velocity_right) / (velocity_right - velocity_left)
    ### END OF STUDENT CODE ####
    ############################
    
    return radius

if __name__ == '__main__':
    print(calculate_radius(.5,0.8))