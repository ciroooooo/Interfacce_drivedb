<xs:schema>
    <xs:element name = "catalogo">
        <xs:complexType>
            <xs:sequence> //sottoelementi con ordine fissato.
                <xs:element name = "libro" minOccurs = "1" maxOccurs = "unbounded"/>
            </xs:sequence>
        </xs:complexType>
    </xs:element>
</xs:schema>

<CatalogoNuovo>{
    for $l in doc("catalogo.xml")/catalogo/libro
    let $titolo := $l/titolo
    let $anno := $l/anno
    WHERE $anno > 2015
    RETURN <Libro><Titolo>{$titolo/text()}</Titolo><Anno>{$anno/text()}</Anno></Libro>
}</CatalogoNuovo>

SELECT ?nome ?titolo
WHERE{
    ?persona rdf:type :Persona .
    ?persona :nome ?nome .
    ?persona :haScritto ?libro .
    ?libro rdf:type :Libro .
    ?libro :titolo ?titolo
}

(p1:Persona)-[:COLLABORA_CON]->(p2:Persona)
(p1)-[:SCRIV]->(l1:Libro)
(p2)-[:SCRIVE]->(l2:Libro)
WHERE p1<>p2 and l1.anno > 2015 and l2.anno>2015
RETURN p1.nome as Persona1, p2.nome as Persona2


<xs:schema>
    <xs:element name = "ordini">
        <xs:complexType>
            <xs:sequence>
                <xs:element name = "ordine">
                    <xs:complextType>
                        <xs:attribute name = "id" type = "xs:integer" use = "required">
                        <xs:sequence>
                            <xs:element name = "data" type = "xs:date" minOccurs = "1" maxOccurs = "1"/>
                            <xs:element name = "cliente" minOccurs = "1" maxOccurs = "1">
                                <xs:complexType>
                                    <xs:sequence>
                                        <xs:element name = "nome" type = "xs:string" minOccurs = "1" maxOccurs = "1"/>
                                        <xs:element name = "email" type = "xs:string" minOccurs = "0" maxOccurs = "1"/>          
                                    <xs:sequence>
                                </xs:complexType>
                        </xs:sequence>
                    </xs:complexType>
                </xs:element>
            </xs:sequence>
        </xs:complexType>
    </xs:element>
</xs:schema>

select a.id, XMLELEMENT(NAME "Articolo",XMLAGG(XMLELEMENT NAME "titolo",a.titolo, XMLELEMENT NAME "anno",a.anno))
FROM Articolo a
GROUP BY a.titolo, a.anno