# scala -Xnojline
// object Game extends App {
    
// }

def bust(hand:Int) : Boolean = {
    hand > 21
}


def bust2(hand:Int) = {
    hand > 21
}

println("Aloha amigo")
println(bust(2))
println(bust2(22))

def maxHand(handA: Int, handB: Int): Int = {
  if (handA > handB) handA
  else handB
}
// ################

// Calculate hand values
// var handPlayerA: Int = queenDiamonds + threeClubs + aceHearts
// var handPlayerB: Int = kingHearts + jackHearts

// // Find and print the maximum hand value
// println(maxHand(handPlayerA, handPlayerB))

// ################

val arreglo = Array("aloha", "pepe")
println(arreglo)

val arr2: Array[String] = new Array[String](3)
println(arr2)
arr2(0) = "Prueba"
println(arr2)

// Create, parameterize, and initialize an array for a round of Twenty-One
val hands3 = Array[Any](tenClubs + fourDiamonds,
              nineSpades + nineHearts,
              twoClubs + threeSpades,
              "Prueba")
val hands = Array[Any](
    tenClubs + fourDiamonds,
    nineSpades + nineHearts,
    twoClubs + threeSpades)

// Initialize player's hand and print out hands before each player hits
hands(0) = tenClubs + fourDiamonds
hands(1) = nineSpades + nineHearts
hands(2) = twoClubs + threeSpades
hands.foreach(println)

// Add 5♣ to the first player's hand
hands(0) = hands(0) + fiveClubs

// Add Q♠ to the second player's hand
hands(1) = hands(1) + queenSpades

// Add K♣ to the third player's hand
hands(2) = hands(2) + kingClubs

// Print out hands after each player hits
hands.foreach(println)

// ####################################
val els = List(1,2,3,4)
val els2 = 4::els // cons operator
val els2_5 = Nil // Nil is empty list
val els3 = "alex"::"perror"::"chen"::Nil // Initialize lists
val els4 = els2:::els4 // ::: concatenation

// Initialize a list with an element for each round's prize
val prizes = 10::15::20::25::30::Nil
println(prizes)

// Prepend to prizes to add another round and prize
val newPrizes = 5::prizes
println(newPrizes)

// The original NTOA and EuroTO venue lists
val venuesNTOA = List("The Grand Ballroom", "Atlantis Casino", "Doug's House")
val venuesEuroTO = "Five Seasons Hotel" :: "The Electric Unicorn" :: Nil

// Concatenate the North American and European venues
val venuesTOWorld = venuesNTOA:::venuesEuroTo