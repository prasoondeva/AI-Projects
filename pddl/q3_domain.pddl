(define (domain pacmanD)
    (:requirements :strips)
    (:types pacman dot loc)
    (:predicates 
        (pacman-at ?p -pacman ?l -loc)
        (DotAt ?d -dot ?x -loc)
        (NotEaten ?d -dot) 
        (east ?x -loc ?y -loc)
        (west ?x -loc ?y -loc)
        (north ?x -loc ?y -loc)
        (south ?x -loc ?y -loc)
    ) 
    
    (:action move-West
        :parameters (?p -pacman ?x -loc ?y -loc)
        :precondition (and (pacman-at ?p ?x) (west ?y ?x) (not (= ?x ?y)))
        :effect (and (not (pacman-at ?p ?x)) (pacman-at ?p ?y)))

    (:action move-East
        :parameters (?p -pacman ?x -loc ?y -loc)
        :precondition (and (pacman-at ?p ?x) (east ?y ?x) (not (= ?x ?y)))
        :effect (and (not (pacman-at ?p ?x)) (pacman-at ?p ?y)))

     (:action move-North
        :parameters (?p -pacman ?x -loc ?y -loc)
        :precondition (and (pacman-at ?p ?x) (north ?y ?x) (not (= ?x ?y)))
        :effect (and (not (pacman-at ?p ?x)) (pacman-at ?p ?y)))

      (:action move-South
        :parameters (?p -pacman ?x -loc ?y -loc)
        :precondition (and (pacman-at ?p ?x) (south ?y ?x) (not (= ?x ?y)))
        :effect (and (not (pacman-at ?p ?x)) (pacman-at ?p ?y)))

      (:action EatDot
        :parameters (?p -pacman ?d -dot ?x -loc)
        :precondition (and (DotAt ?d ?x) (pacman-at ?p ?x) (NotEaten ?d))
        :effect (and (not (DotAt ?d ?x)) (not (NotEaten ?d))))
)