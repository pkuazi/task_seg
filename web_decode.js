var GSON = {
    base_number : 100000.0,
    base_offset : 64,
    decode_a_line : function(coordinate, encodeOffsets) {
        var result = [];

        var prevX = encodeOffsets[0]
        var prevY = encodeOffsets[1]

        var l = coordinate.length;
        var i = 0;
        while (i < l) {
            var x = coordinate.charCodeAt(i) - this.base_offset
            var y = coordinate.charCodeAt(i + 1) - this.base_offset
            i += 2;

            x = (x >> 1) ^ (-(x & 1));
            y = (y >> 1) ^ (-(y & 1));

            x += prevX;
            y += prevY;

            prevX = x;
            prevY = y;

            result.push([ x / this.base_number, y / this.base_number ])
        }
        return result;
    }, 

    decode_geojson: function(geojson) {
        var gtype = geojson["type"];
        var coordinates = geojson["coordinates"];       

        var new_coordinates = [];        
        if( gtype == "MultiPolygon" ){ 
            for (var i = 0; i < coordinates.length; i++) {
                var a_polygon = coordinates[i];
                a_polygon = a_polygon[0];
                var aline = this.decode_a_line(a_polygon[0], a_polygon[1]);
                new_coordinates.push([ aline ])
            } 
        }else if ( gtype == "Polygon" || qtype == "MultiLineString" ){
                for (var i = 0; i < coordinates.length; i++) {
                    var a_polygon = coordinates[i];
                    a_polygon = a_polygon[0];
                    var aline = this.decode_a_line(a_polygon[0], a_polygon[1]);
                    new_coordinates.push( aline );
                } 
        } else if ( gtype == "LineString" ){ 
            var coordinates = coordinates[0];
            new_coordinates = this.decode_a_line(a_polygon[0], a_polygon[1]);
        } 

        geojson["coordinates"] = new_coordinates ;
        return geojson  ;
    }
};

