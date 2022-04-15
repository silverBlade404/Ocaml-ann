open Printf                       
type 'a io       = { i: 'a; o: 'a }                                  
type vec         = float array          
type mat         = vec array              
type neuralNet   = { a : vec io; ah : vec; w : mat io; c : mat io }          
let vector       = Array.init   
          
let matrix m n f = vector m (fun i -> vector n (f i))                        

let neuralNet ni nh no =                  
    let init fi fo = { i = matrix (ni + 1) nh fi; o = matrix nh no fo } in   
    let rand x0 x1 = x0 +. Random.float(x1 -. x0) in                         
    { 
      a = { i = vector (ni + 1) (fun _ -> 1.0); o = vector no (fun _ -> 1.0) };     
      ah = vector nh (fun _ -> 1.0);      
      w = init (fun _ _ -> rand (-0.2) 0.4) (fun _ _ -> rand (-0.2) 0.4);    
      c = init (fun _ _ -> 0.0) (fun _ _ -> 0.0)                             
    }  

let sigmoid x = 1.0 /. (1.0 +. exp(-. x)) 

let sigmoid' y = y *. (1.0 -. y)          

let rec fold2 n f a xs ys =               
    let a = ref a in                      
    for i=0 to n-1 do                     
        a := f !a (xs i) (ys i)           
    done;                                 
    !a 

let dot n xs ys = fold2 n (fun t x y -> t +. x *. y) 0.0 xs ys               
let length      = Array.length            
let get         = Array.get               

let forward net inputs =                   
    let ni, nh, no = Array.length net.a.i, Array.length net.ah, Array.length net.a.o in        
    assert(length inputs = ni-1);         
    let ai i = if i < ni-1 then inputs.(i) else net.a.i.(i) in               
    let ah j = sigmoid(dot ni ai (fun i -> net.w.i.(i).(j))) in              
    let ah   = vector nh ah in            
    let ao k = sigmoid(dot nh (Array.get ah) (fun j -> net.w.o.(j).(k))) in        
    {net with a = { i = vector ni ai; o = vector no ao }; ah = ah }          



let rec train net patterns iters n m =    
    if iters = 0 then net else            
        let step (net, error) (inputs, targets) =                            
            let net, de = backPropagate (forward net inputs) targets n m in   
            net, error +. de in           
        let net, error = Array.fold_left step (net, 0.0) patterns in         
        if iters mod 10000 = 0 then printf "Error: %g:\n%!" error;           
        train net patterns (iters - 1) n m
               

let df =                               
    [|[|2.0; 0.0|] , [|2.0|];             
      [|4.0; 1.0|] , [|1.0|];             
      [|1.0; 5.0|] , [|1.0|];             
      [|1.0; -1.0|] , [|4.0|]|]            

let () =                                  
    let t = Sys.time() in                 
    let net = neuralNet 2 2 1 in          
    test df (train net df 100000 0.5 0.1);                             
    printf "Took %gs\n" (Sys.time() -. t)