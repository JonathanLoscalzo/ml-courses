var request = require("request");
var encodedPat = encodePat('jloscalzo:JRloscalzo@2');
 
var options = {
   method: 'GET',
   headers: { 'cache-control': 'no-cache', 'authorization': `Basic ${encodedPat}` },
   url: 'https://tfs.hexacta.com/Erebor/_apis/wit/workitems/1234?api-version=2.3',
   qs: { 'api-version': '2.3' }
};
 
request(options, function (error, response, body) {
  if (error) throw new Error(error);
 
  console.log(body);
});
 
function encodePat(pat) {
   var b = new Buffer(':' + pat);
   var s = b.toString('base64');
    console.log(b, s)
   return s;
}