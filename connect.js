const express = require("express")
const {Web3} = require("web3")
const api = "2654cc40f5f74188b0018adc2994171a"
const web3 = new Web3(new Web3.providers.HttpProvider("https://sepolia.infura.io/v3/2654cc40f5f74188b0018adc2994171a"))
web3.eth.getBlockNumber().then(blockNumber => {console.log(blockNumber)})

