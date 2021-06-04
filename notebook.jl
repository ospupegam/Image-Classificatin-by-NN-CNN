### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 2752c630-78b3-40ad-9416-eee1275179b2
begin
	using Printf
	using Pkg
	Pkg.add("Flux")
	Pkg.add("MLJ")
	Pkg.add("MLDatasets")
	Pkg.add("IterTools")
	Pkg.add("PlutoUI")
	Pkg.add("Images")
	Pkg.add("StatsBase")
	using Flux, PlutoUI, Statistics, MLDatasets, Images
	using Flux.Data: DataLoader 
	using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, throttle, unsqueeze
	using Random: shuffle
	using StatsBase: countmap, proportionmap
	using IterTools
	using MLJ
end

# ╔═╡ fba5a130-c309-11eb-1a74-1d240c4581a5


# ╔═╡ 7b858214-e134-4009-8f53-69428dc8902f
md"# Fashion MNIST training"

# ╔═╡ 707152fb-5f7c-486e-80fd-28ff64a87245
md""" If you are running this for first time, you will need to download the 
data first This is done by 
```julia
	FashionMNIST.download(;i_accept_the_terms_of_use=true);
```

You can examin the full documentaiton [here](https://juliaml.github.io/MLDatasets.jl/latest/datasets/FashionMNIST/). 
"""

# ╔═╡ da44348b-e102-4595-9b9e-3607680ba7e1
 FashionMNIST.download(;i_accept_the_terms_of_use=true);

# ╔═╡ fbcd01ab-8c89-424f-a890-6a8b0d26354c
train_x, train_y = MLDatasets.FashionMNIST.traindata(Float32);

# ╔═╡ 5dfc00ee-1a1a-4d10-a55e-94f3dd882a89
test_x,  test_y  = MLDatasets.FashionMNIST.testdata();

# ╔═╡ 49b650e9-3ac4-48a0-bda2-05e4c7cf0cc0
Y_train, Y_test = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9);

# ╔═╡ 4420d2aa-2159-451c-ac08-81cb220e8451
md"We can look at few samples. Note that I do `1. .- tensor` to change the background to white"

# ╔═╡ 39dd4d30-4096-4738-943b-b599433405d4
FashionMNIST.convert2image(1. .- test_x[:,:,6])

# ╔═╡ 6efeb660-7e62-479c-bd2a-572ef570dbc2
FashionMNIST.convert2image(1. .- train_x[:,:,6])

# ╔═╡ faebd51d-b20f-429b-995e-0172750dfb07
begin
	image_url = "https://pravarmahajan.github.io/assets/images/fashion-MNIST/labels_table.png"
	md"""
	$(Resource(image_url, :width => 100))"""
	
end

# ╔═╡ 82295eba-144e-4672-8255-cda02e68016a
begin
	class_dic = Dict(
		0 => "T-shirt/top", 
		1 => "Trouser", 
		2 => "Pullover", 
		3 => "Dress", 
		4 => "Coat", 
		5 => "Sandal", 
		6 => "Shirt", 
		7 => "Sneaker", 
		8 => "Bag", 
		9 => "Ankle boot"
	) 
	md"I use the above table to construct a dictionary to make labels to descriptions. Stored in the `class_dic` variable"
end

# ╔═╡ 24bd879f-c2e9-4700-b07b-c850b4856058
md"""
## Assignment Questions 


!!! question "Question 1 (4 pts)"

    Using Flux `Chain` build a Dense MLP NN, with following layers in sequence 784, 300, 100, 10. Run the output of the last layer through a `softmax` function.     


# ╔═╡ f738066e-d939-4f6a-ac6f-d35232ae7888
X_train = Flux.flatten(train_x);    

# ╔═╡ da6fd517-ecb9-4600-9a8f-32db6d2ae130
X_test = Flux.flatten(test_x);

# ╔═╡ 3a055e5a-737f-47d5-b076-6cf5ab5e6048
model = Chain(
	Flux.Dense(784,300,relu), # Flattened input Image -> 300 Nodes Hidden Layer 1
	Flux.Dense(300,100,relu), # 300 Nodes Hidden Layer 1-> 100 Nodes Hidden Layer 2
	Flux.Dense(100,10), # 100 Nodes Hidden Layer 2-> 10 Nodes Output
	softmax # Softmax for Classififation
);

# ╔═╡ 3c3cd076-fad1-486a-a33f-b7a79c227882
#!!! question "Question 2 (2 pts)"

 #   a) Define a loss funtion based on `crossentropty`

# ╔═╡ 8b37d375-cd30-4290-92c0-5a58dd53e393
loss(x,y) = Flux.Losses.crossentropy(model(x),y) 

# ╔═╡ d487fbbd-2f39-47f0-8343-bed56ea5cdf9
#b) Instatiate an `ADAM` optimizer 
opt = Flux.Optimise.ADAM();

# ╔═╡ 9f958eb9-4bd9-4263-9e9f-57108a07a4ef
#c) Define an `accuracy` function that could runn over the whole dataset (Hint: #make use of `onecold` and `mean`)

accuracy(x,y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y));

# ╔═╡ 846ae5c4-feff-4104-8845-c882f72e3f3e
#!!! question "Question 3 (3 pts)"

 #   Define a funciton `badIdx` that filter for the indices where the model failes to #classify correctly. 


# ╔═╡ afbe3cc0-b5c3-4867-86bb-08bab9aa11d4


# ╔═╡ 1f043f46-ab9c-49c0-bb68-6ac1b228a377
#!!! question "Question 4 (3 pts)"

 #   Make use of `nycle` and `DataLoader` and `Flux.train!` to train the model with `batchsize` of 50 and 10 epochs (Hint: see [docs](https://fluxml.ai/Flux.jl/stable/data/dataloader/)). Use a callback function, `cb` to the `accuracy` and `loss` every 10 seconds.  


# ╔═╡ 809d9fe5-a334-4d6a-8ee7-746ddbfbaaba
begin
	train_loader = Flux.Data.DataLoader((X_train, Y_train), batchsize=50, shuffle=true)
end

# ╔═╡ 95cca89c-261c-4576-94c0-785824257b24
parameters = Flux.params(model)

# ╔═╡ af1a2aca-cce4-46b4-bd42-2cd5fdbdb48a
evalcbal() = @show(loss(X_test, Y_test),accuracy(X_test, Y_test))

# ╔═╡ 6cd8c3b9-8710-44fc-9def-9ee36d7288e3

Flux.Optimise.@epochs 10 Flux.Optimise.train!(loss, parameters, train_loader, opt, cb = Flux.throttle(evalcbal, 10))

# ╔═╡ 38228045-0d69-4b8f-a2db-5051d668bc59
accuracy(X_test,Y_test)

# ╔═╡ 87173fb2-4186-4380-8322-e3b51fc2d828
#!!! question "Question 5 (4 pts)"

 #   Which top 3 classes did the Dense MLP NN model struggle with the most? Are ther #error rates uniform across the classes? 


# ╔═╡ 81e4227d-dc59-4cfd-9a8d-334ad901aab2
prediction, actual = Flux.onecold(model(X_test[:,5]), 0:9), Flux.onecold(Y_test[:,5],0:9)

# ╔═╡ a0421745-d89e-47f3-9a42-e07cceac5ba5
#!!! question "Question 6 (3 pts)"

 #   Build you own *custom training loop* with same parameters. Check that it works #just like in question 4.

# ╔═╡ 4eb122e7-74f2-4687-a0ea-40970706c029


# ╔═╡ 67cd1972-27c7-4aa4-910f-9d21a664b7cf
#!!! question "Question 7 (5 pts)"

 #   Construct a convolutional model with following architecture 

#		1. Conv with a (5,5) kernel mapping to **6** feature maps, with `relu`,  same paddig 
#		2. 2x2 Max pool 
#		3. Conv with a (5,5) kernel mapping to **16** feature maps, with `relu`,  same paddig 
#		4. 2x2 Max pool
#		5. Appropriatly sized Dense layer with 10 outputs
#		6. A `softmax` layer
#	Do the training using a *custom training loop* 

# ╔═╡ 294453cf-16b4-4d2b-aeea-f87bed8902ec
mySimpleChain1 = Chain(
	Conv((5, 5), 1=>6, relu),
	MaxPool((2, 2)),
	Conv((5, 5), 6=>16, relu),
	MaxPool((2, 2)),
	flatten,
	Dense(prod(256),300,relu),
	Dense(300,100,relu),
	Dense(100,10),
	softmax
)

# ╔═╡ c4d391bd-3815-4c33-88f3-abf2fb044cf7
mySimpleChain1(rand(28,28,1,1))

# ╔═╡ 50f04a5b-3d85-4ebd-b54a-108c3877cf51
Flux.params(mySimpleChain1);

# ╔═╡ ac2f9393-96be-4936-9ec8-699c562a5609
#!!! question "Question 8 (3 pts)"

#	Do changes in the convnet  architecture to get a better accuracy. 


# ╔═╡ f57ca442-d19d-4612-b828-39b9be320088

#!!! question "Question 9 (3 pts)"

#	Did the top 3 classes in Question 6 change with the use of the networks in Questions 8 or 9?  










"""

# ╔═╡ Cell order:
# ╟─9ddb04c6-6831-458e-a1ea-da0e5a079d9a
# ╠═99ee4fa6-a080-11eb-1df7-1b2a6abf52a7
# ╟─8680b6e8-f540-4763-94b4-2c55f90e1f60
# ╠═1065d8b4-a1a5-473b-b12e-79d103ef2d09
# ╠═654be88c-dcee-426e-88ad-72ca6e6263f0
# ╠═07554410-7fc7-4099-926a-b518fe1f833d
# ╟─b03a940e-cf96-4d25-a398-679bb36b9c37
# ╠═23245d5e-db7d-484a-8a42-50176832a058
# ╠═2086201c-c6da-408e-a865-3e49da23c263
# ╟─102f57c2-3af7-424f-933c-ea4ad9f77593
# ╟─23214778-6693-4124-8e6d-5df995644dbe
# ╟─df26309b-bb5a-4af8-98b6-fa1bf3accd8d

# ╔═╡ Cell order:
# ╠═fba5a130-c309-11eb-1a74-1d240c4581a5
# ╠═2752c630-78b3-40ad-9416-eee1275179b2
# ╠═7b858214-e134-4009-8f53-69428dc8902f
# ╠═707152fb-5f7c-486e-80fd-28ff64a87245
# ╠═da44348b-e102-4595-9b9e-3607680ba7e1
# ╠═fbcd01ab-8c89-424f-a890-6a8b0d26354c
# ╠═5dfc00ee-1a1a-4d10-a55e-94f3dd882a89
# ╠═49b650e9-3ac4-48a0-bda2-05e4c7cf0cc0
# ╠═4420d2aa-2159-451c-ac08-81cb220e8451
# ╠═39dd4d30-4096-4738-943b-b599433405d4
# ╠═6efeb660-7e62-479c-bd2a-572ef570dbc2
# ╠═faebd51d-b20f-429b-995e-0172750dfb07
# ╠═82295eba-144e-4672-8255-cda02e68016a
# ╠═24bd879f-c2e9-4700-b07b-c850b4856058
# ╠═f738066e-d939-4f6a-ac6f-d35232ae7888
# ╠═da6fd517-ecb9-4600-9a8f-32db6d2ae130
# ╠═3a055e5a-737f-47d5-b076-6cf5ab5e6048
# ╠═3c3cd076-fad1-486a-a33f-b7a79c227882
# ╠═8b37d375-cd30-4290-92c0-5a58dd53e393
# ╠═d487fbbd-2f39-47f0-8343-bed56ea5cdf9
# ╠═9f958eb9-4bd9-4263-9e9f-57108a07a4ef
# ╠═846ae5c4-feff-4104-8845-c882f72e3f3e
# ╠═afbe3cc0-b5c3-4867-86bb-08bab9aa11d4
# ╠═1f043f46-ab9c-49c0-bb68-6ac1b228a377
# ╠═809d9fe5-a334-4d6a-8ee7-746ddbfbaaba
# ╠═95cca89c-261c-4576-94c0-785824257b24
# ╠═af1a2aca-cce4-46b4-bd42-2cd5fdbdb48a
# ╠═6cd8c3b9-8710-44fc-9def-9ee36d7288e3
# ╠═38228045-0d69-4b8f-a2db-5051d668bc59
# ╠═87173fb2-4186-4380-8322-e3b51fc2d828
# ╠═81e4227d-dc59-4cfd-9a8d-334ad901aab2
# ╠═a0421745-d89e-47f3-9a42-e07cceac5ba5
# ╠═4eb122e7-74f2-4687-a0ea-40970706c029
# ╠═67cd1972-27c7-4aa4-910f-9d21a664b7cf
# ╠═294453cf-16b4-4d2b-aeea-f87bed8902ec
# ╠═c4d391bd-3815-4c33-88f3-abf2fb044cf7
# ╠═50f04a5b-3d85-4ebd-b54a-108c3877cf51
# ╠═ac2f9393-96be-4936-9ec8-699c562a5609
# ╠═f57ca442-d19d-4612-b828-39b9be320088
